# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import cast
import math
import os

import numpy as np
import torch
from torch.distributions import Normal
from torch_geometric.data.batch import Batch

from .smc_utils import resampling_function, normalize_log_weights

from .chemgraph import ChemGraph
from .sde_lib import SDE, CosineVPSDE
from .so3_sde import SO3SDE, apply_rotvec_to_rotmat, update_rotation_matrix_from_gradient, rotmat_to_rotvec, weighted_rigid_align

from collections import defaultdict


TwoBatches = tuple[Batch, Batch]

def log_normal_density(sample, mean, var):
    return Normal(loc=mean, scale=torch.sqrt(var)).log_prob(sample)

class EulerMaruyamaPredictor:
    """Euler-Maruyama predictor."""

    def __init__(
        self,
        *,
        corruption: SDE,
        noise_weight: float = 1.0,
        marginal_concentration_factor: float = 1.0,
    ):
        """
        Args:
            noise_weight: A scalar factor applied to the noise during each update. The parameter controls the stochasticity of the integrator. A value of 1.0 is the
            standard Euler Maruyama integration scheme whilst a value of 0.0 is the probability flow ODE.
            marginal_concentration_factor: A scalar factor that controls the concentration of the sampled data distribution. The sampler targets p(x)^{MCF} where p(x)
            is the data distribution. A value of 1.0 is the standard Euler Maruyama / probability flow ODE integration.

            See feynman/projects/diffusion/sampling/samplers_readme.md for more details.

        """
        self.corruption = corruption
        self.noise_weight = noise_weight
        self.marginal_concentration_factor = marginal_concentration_factor

    def reverse_drift_and_diffusion(
        self, *, x: torch.Tensor, t: torch.Tensor, batch_idx: torch.LongTensor, score: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        score_weight = 0.5 * self.marginal_concentration_factor * (1 + self.noise_weight**2)
        # drift = -1/2 beta_t x (appropriately broadcast)
        # diffusion = sqrt(beta_t)
        drift, diffusion = self.corruption.sde(x=x, t=t, batch_idx=batch_idx)
        drift = drift - diffusion**2 * score * score_weight
        return drift, diffusion

    def update_given_drift_and_diffusion(
        self,
        *,
        x: torch.Tensor,
        dt: torch.Tensor,
        drift: torch.Tensor,
        diffusion: torch.Tensor,
    ) -> TwoBatches:
        z = torch.randn_like(drift)
        diffusion_step = self.noise_weight * diffusion * torch.sqrt(dt.abs()) * z

        # Update to next step using either special update for SDEs on SO(3) or standard update.
        if isinstance(self.corruption, SO3SDE):
            mean = apply_rotvec_to_rotmat(x, drift * dt, tol=self.corruption.tol)
            sample = apply_rotvec_to_rotmat(
                mean,
                diffusion_step,
                tol=self.corruption.tol,
            )
        else:
            mean = x + drift * dt
            sample = mean + diffusion_step
        return sample, mean, diffusion_step

    def update_given_score(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
        score: torch.Tensor,
    ) -> TwoBatches:

        # Set up different coefficients and terms.
        drift, diffusion = self.reverse_drift_and_diffusion(
            x=x, t=t, batch_idx=batch_idx, score=score
        )

        # Update to next step using either special update for SDEs on SO(3) or standard update.
        return self.update_given_drift_and_diffusion(
            x=x,
            dt=dt,
            drift=drift,
            diffusion=diffusion,
        )

    def forward_sde_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
    ) -> TwoBatches:
        """Update to next step using either special update for SDEs on SO(3) or standard update.
        Handles both SO(3) and Euclidean updates."""

        drift, diffusion = self.corruption.sde(x=x, t=t, batch_idx=batch_idx)
        # Update to next step using either special update for SDEs on SO(3) or standard update.
        return self.update_given_drift_and_diffusion(x=x, dt=dt, drift=drift, diffusion=diffusion)


def _get_score(
    batch: ChemGraph, sdes: dict[str, SDE], score_model: torch.nn.Module, t: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Calculate predicted score for the batch.

    Args:
        batch: Batch of corrupted data.
        sdes: SDEs.
        score_model: Score model.  The score model is parametrized to predict a multiple of the score.
          This function converts the score model output to a score.
        t: Diffusion timestep. Shape [batch_size,]
    """
    batch.node_orientations = batch.node_orientations.requires_grad_(True)

    tmp = score_model(batch, t)

    # Score is in axis angle representation [N,3] (vector is along axis of rotation, vector length
    # is rotation angle in radians).
    assert isinstance(sdes["node_orientations"], SO3SDE)
    node_orientations_score = (
        tmp["node_orientations"]
        * sdes["node_orientations"].get_score_scaling(t, batch_idx=batch.batch)[:, None]
    )

    # Score model is trained to predict score * std, so divide by std to get the score.
    _, pos_std = sdes["pos"].marginal_prob(
        x=torch.ones_like(tmp["pos"]),
        t=t,
        batch_idx=batch.batch,
    )
    pos_score = tmp["pos"] / pos_std


    return {"node_orientations": node_orientations_score, "pos": pos_score}

def dpm_solver(
    sdes: dict[str, SDE],
    batch: Batch,
    N: int,
    score_model: torch.nn.Module,
    max_t: float,
    eps_t: float,
    device: torch.device,
    **kwargs
) -> tuple[ChemGraph, ChemGraph, list[ChemGraph] | None, list[ChemGraph] | None]:

    """
    Implements the DPM solver for the VPSDE, with the Cosine noise schedule.
    Following this paper: https://arxiv.org/abs/2206.00927 Algorithm 1 DPM-Solver-2.
    DPM solver is used only for positions, not node orientations.
    """
    assert isinstance(batch, ChemGraph)
    assert max_t < 1.0

    batch = batch.to(device)
    if isinstance(score_model, torch.nn.Module):
        # permits unit-testing with dummy model
        score_model = score_model.to(device)
    pos_sde = sdes["pos"]
    assert isinstance(pos_sde, CosineVPSDE)

    batch = batch.replace(
        pos=sdes["pos"].prior_sampling(batch.pos.shape, device=device),
        node_orientations=sdes["node_orientations"].prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )
    batch = cast(ChemGraph, batch)  # help out mypy/linter

    so3_sde = sdes["node_orientations"]
    assert isinstance(so3_sde, SO3SDE)
    so3_sde.to(device)

    timesteps = torch.linspace(max_t, eps_t, N, device=device)
    dt = -torch.tensor((max_t - eps_t) / (N - 1)).to(device)

    for i in range(N - 1):
        t = torch.full((batch.num_graphs,), timesteps[i], device=device)

        # Evaluate score
        score = _get_score(batch=batch, t=t, score_model=score_model, sdes=sdes)
        # t_{i-1} in the algorithm is the current t
        batch_idx = batch.batch
        alpha_t, sigma_t = pos_sde.mean_coeff_and_std(x=batch.pos, t=t, batch_idx=batch_idx)
        lambda_t = torch.log(alpha_t / sigma_t)
        alpha_t_next, sigma_t_next = pos_sde.mean_coeff_and_std(
            x=batch.pos, t=t + dt, batch_idx=batch_idx
        )
        lambda_t_next = torch.log(alpha_t_next / sigma_t_next)

        # t+dt < t, lambad_t_next > lambda_t
        h_t = lambda_t_next - lambda_t

        # For a given noise schedule (cosine is what we use), compute the intermediate t_lambda
        lambda_t_middle = (lambda_t + lambda_t_next) / 2
        t_lambda = _t_from_lambda(sde=pos_sde, lambda_t=lambda_t_middle)

        # t_lambda has all the same components
        t_lambda = torch.full((batch.num_graphs,), t_lambda[0][0], device=device)

        alpha_t_lambda, sigma_t_lambda = pos_sde.mean_coeff_and_std(
            x=batch.pos, t=t_lambda, batch_idx=batch_idx
        )
        # Note in the paper the algorithm uses noise instead of score, but we use score.
        # So the formulation is slightly different in the prefactor.
        u = (
            alpha_t_lambda / alpha_t * batch.pos
            + sigma_t_lambda * sigma_t * (torch.exp(h_t / 2) - 1) * score["pos"]
        )

        # Update positions to the intermediate timestep t_lambda
        batch_u = batch.replace(pos=u)

        # Get node orientation at t_lambda

        # Denoise from t to t_lambda
        assert score["node_orientations"].shape == (u.shape[0], 3)
        assert batch.node_orientations.shape == (u.shape[0], 3, 3)
        so3_predictor = EulerMaruyamaPredictor(
            corruption=so3_sde, noise_weight=0.0, marginal_concentration_factor=1.0
        )
        drift, _ = so3_predictor.reverse_drift_and_diffusion(
            x=batch.node_orientations,
            score=score["node_orientations"],
            t=t,
            batch_idx=batch_idx,
        )
        sample, _, _ = so3_predictor.update_given_drift_and_diffusion(
            x=batch.node_orientations,
            drift=drift,
            diffusion=0.0,
            dt=t_lambda[0] - t[0],
        )  # dt is negative, diffusion is 0
        assert sample.shape == (u.shape[0], 3, 3)
        batch_u = batch_u.replace(node_orientations=sample)

        # Correction step
        # Evaluate score at updated pos and node orientations
        score_u = _get_score(batch=batch_u, t=t_lambda, sdes=sdes, score_model=score_model)

        pos_next = (
            alpha_t_next / alpha_t * batch.pos
            + sigma_t_next * sigma_t_lambda * (torch.exp(h_t) - 1) * score_u["pos"]
        )

        batch_next = batch.replace(pos=pos_next)

        assert score_u["node_orientations"].shape == (u.shape[0], 3)

        # Try a 2nd order correction
        node_score = (
            score_u["node_orientations"]
            + 0.5
            * (score_u["node_orientations"] - score["node_orientations"])
            / (t_lambda[0] - t[0])
            * dt
        )
        drift, _ = so3_predictor.reverse_drift_and_diffusion(
            x=batch_u.node_orientations,
            score=node_score,
            t=t_lambda,
            batch_idx=batch_idx,
        )
        sample, _, _ = so3_predictor.update_given_drift_and_diffusion(
            x=batch.node_orientations,
            drift=drift,
            diffusion=0.0,
            dt=dt,
        )  # dt is negative, diffusion is 0
        batch = batch_next.replace(node_orientations=sample)

    return batch

def first_order_denoiser(
    *,
    sdes: dict[str, SDE],
    N: int,
    eps_t: float,
    max_t: float,
    device: torch.device,
    batch: Batch,
    score_model: torch.nn.Module,
    noise: float,
    sequence: str,
    **kwargs
) -> ChemGraph:
    """Sample from prior and then denoise."""

    batch = batch.to(device)
    if isinstance(score_model, torch.nn.Module):
        # permits unit-testing with dummy model
        score_model = score_model.to(device)
    assert isinstance(sdes["node_orientations"], torch.nn.Module)  # shut up mypy
    sdes["node_orientations"] = sdes["node_orientations"].to(device)
    batch = batch.replace(
        pos=sdes["pos"].prior_sampling(batch.pos.shape, device=device),
        node_orientations=sdes["node_orientations"].prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )

    ts_min = 0.0
    ts_max = 1.0
    timesteps = torch.linspace(max_t, eps_t, N, device=device)
    dt = -torch.tensor((max_t - eps_t) / (N - 1)).to(device)
    fields = list(sdes.keys())
    predictors = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=0.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }
    noisers = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=1.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }
    batch_size = batch.num_graphs
    

    for i in range(N):
        # Set the timestep
        t = torch.full((batch_size,), timesteps[i], device=device)
        t_next = t + dt  # dt is negative; t_next is slightly less noisy than t.

        # Select temporarily increased noise level t_hat.
        # To be more general than Algorithm 2 in Karras et al. we select a time step between the
        # current and the previous t.
        t_hat = t - noise * dt if (i > 0 and t[0] > ts_min and t[0] < ts_max) else t

        # Apply noise.
        vals_hat = {}
        for field in fields:
            vals_hat[field] = noisers[field].forward_sde_step(
                x=batch[field], t=t, dt=(t_hat - t)[0], batch_idx=batch.batch
            )[0]
        batch_hat = batch.replace(**vals_hat)

        score = _get_score(batch=batch_hat, t=t_hat, score_model=score_model, sdes=sdes)

        # First-order denoising step from t_hat to t_next.
        drift_hat = {}
        for field in fields:
            drift_hat[field], _ = predictors[field].reverse_drift_and_diffusion(
                x=batch_hat[field], t=t_hat, batch_idx=batch.batch, score=score[field]
            )

        for field in fields:
            batch[field] = predictors[field].update_given_drift_and_diffusion(
                x=batch_hat[field],
                dt=(t_next - t_hat)[0],
                drift=drift_hat[field],
                diffusion=0.0,
            )[0]

    return batch


def _checkpointed_step_function(
    batch, t, t_next, t_hat, dt, noise, fields, sdes, score_model, predictors, noisers
):
    """
    This function executes a single step of the denoising loop.
    It's designed to be wrapped by torch.utils.checkpoint.
    """
    vals_hat = {}
    for field in fields:
        vals_hat[field] = noisers[field].forward_sde_step(
            x=batch[field], t=t, dt=(t_hat - t)[0], batch_idx=batch.batch
        )[0]
    batch_hat = batch.replace(**vals_hat)
    score = _get_score(batch=batch_hat, t=t_hat, score_model=score_model, sdes=sdes)
    drift_hat = {}
    for field in fields:
        drift_hat[field], _ = predictors[field].reverse_drift_and_diffusion(
            x=batch_hat[field], t=t_hat, batch_idx=batch.batch, score=score[field]
        )
    next_batch_vals = {}
    for field in fields:
        next_batch_vals[field] = predictors[field].update_given_drift_and_diffusion(
            x=batch_hat[field],
            dt=(t_next - t_hat)[0],
            drift=drift_hat[field],
            diffusion=0.0,
        )[0]
    return batch.replace(**next_batch_vals)


def first_order_denoiser_from_intermediate_ckpt(
    *,
    sdes: dict[str, SDE],
    N: int,
    eps_t: float,
    max_t: float,
    device: torch.device,
    batch: Batch,
    score_model: torch.nn.Module,
    noise: float,
    sequence: str,
    **kwargs
) -> ChemGraph:
    """Starting from intermediates and then denoise, WITH GRADIENT CHECKPOINTING."""

    print(f"calling intermediate denoiser N={N} eps_t={eps_t} max_t={max_t} WITH CHECKPOINTING")

    batch = batch.to(device)
    if isinstance(score_model, torch.nn.Module):
        score_model = score_model.to(device)
    assert isinstance(sdes["node_orientations"], torch.nn.Module)
    sdes["node_orientations"] = sdes["node_orientations"].to(device)

    timesteps = torch.linspace(max_t, eps_t, N, device=device)
    dt = -torch.tensor((max_t - eps_t) / (N - 1)).to(device)
    fields = list(sdes.keys())
    predictors = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=0.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }
    noisers = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=1.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }
    batch_size = batch.num_graphs

    for i in range(N):
        t = torch.full((batch_size,), timesteps[i], device=device)
        t_next = t + dt
        t_hat = t - noise * dt if i > 0 else t

        batch = torch.utils.checkpoint.checkpoint(
            _checkpointed_step_function,
            batch, t, t_next, t_hat, dt, noise, fields, sdes, score_model, predictors, noisers,
            use_reentrant=False,
            preserve_rng_state=True 
        )

    return batch

def first_order_denoiser_tds(
    *,
    sdes: dict[str, SDE],
    N: int,
    eps_t: float,
    max_t: float,
    device: torch.device,
    batch: Batch,
    score_model: torch.nn.Module,
    noise: float,
    sequence: str,
    beta: float, # distance in nm to twist to
    c0: int = None, # center of first twist region
    c1: int = None, # center of second twist region
    twist_rmsd: float = False, # Whether to twist with respect to RMSD
    ss_mask: torch.Tensor = None, # mask for which atoms to twist
    untwisted_coords: torch.Tensor = None, # coordinates of the untwisted structure
    twist_k: float, # coefficient for twisting
    extra_twist_k: float, # coefficient for twisting
    enable_guidance: bool = True, # whether to enable guidance
    resample_start: int = 20, # what timestep to start resampling at
    **kwargs
) -> ChemGraph:
    """Sample from prior and then denoise."""

    batch = batch.to(device)
    if isinstance(score_model, torch.nn.Module):
        # permits unit-testing with dummy model
        score_model = score_model.to(device)
    assert isinstance(sdes["node_orientations"], torch.nn.Module)  # shut up mypy
    sdes["node_orientations"] = sdes["node_orientations"].to(device)
    batch = batch.replace(
        pos=sdes["pos"].prior_sampling(batch.pos.shape, device=device),
        node_orientations=sdes["node_orientations"].prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )

    ts_min = 0.0
    ts_max = 1.0
    timesteps = torch.linspace(max_t, eps_t, N, device=device)
    dt = -torch.tensor((max_t - eps_t) / (N - 1)).to(device)
    fields = list(sdes.keys())
    predictors = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=0.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }
    noisers = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=1.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }
    batch_size = batch.num_graphs

    # for logging
    ddr_final_ests = [] # 2nd order estimate of t_final guidance value, pre guidance
    ddr_curr_ests = [] # t_now guidance value
    ddr_post_guidance_ests = [] # t_now guidance value post guidance
    ddr_post_guidance_final_ests = [] #  2nd order estimate of t_final guidance value, post guidance
    

    guidance_delta = torch.zeros_like(batch['pos'])
    guidance_prob = prior_guidance = torch.zeros(batch_size, device=device)
    guidance_prob_first = None
    guidance_performed = False # keep track every iteration
    log_normal_xtm1 = log_normal_xt = None
    log_wtp1 = torch.zeros_like(guidance_prob)

    if ss_mask is not None:
        ss_mask_region = ss_mask.to(device)
        untwisted_pos = untwisted_coords.to(device)

    for i in range(N):
        # Set the timestep
        t = torch.full((batch_size,), timesteps[i], device=device)
        t_next = t + dt  # dt is negative; t_next is slightly less noisy than t.

        # Select temporarily increased noise level t_hat.
        # To be more general than Algorithm 2 in Karras et al. we select a time step between the
        # current and the previous t.
        t_hat = t - noise * dt if (i > 0 and t[0] > ts_min and t[0] < ts_max) else t


        # Apply noise.
        vals_hat = {}
        means_hat = {}
        diffusion_steps_hat = {}
        for field in fields:
            vals_hat[field], means_hat[field], diffusion_steps_hat[field] = noisers[field].forward_sde_step(
                x=batch[field], t=t, dt=(t_hat - t)[0], batch_idx=batch.batch
            )
        batch_hat = batch.replace(**vals_hat)

        if guidance_performed and batch_size > 1:
            # we have the info we need to do resampling
            noise_dt = (t_hat - t)[0]
            pos_drift, pos_diffusion = noisers['pos'].corruption.sde(batch[field], t=t, batch_idx=batch.batch)

            # log prob of where we ended up, given that we performed guidance
            log_prob_xt_given_y_xtm1_pos = Normal(
                    loc=means_hat['pos'],
                    scale=noisers['pos'].noise_weight * pos_diffusion * torch.sqrt(noise_dt.abs())
                ).log_prob(vals_hat['pos'])

            node_orientations_drift, node_orientations_diffusion = noisers['node_orientations'].corruption.sde(batch[field], t=t, batch_idx=batch.batch)

            log_prob_xt_given_y_xtm1_node_orientations = Normal(
                    loc=torch.zeros_like(diffusion_steps_hat['node_orientations']),
                    scale=(noisers['node_orientations'].noise_weight * node_orientations_diffusion * torch.sqrt(noise_dt.abs()))[:, None]
                ).log_prob(diffusion_steps_hat['node_orientations'])

            # log prob of where we would have ended up, had we not performed guidance

            # the drift to t_hat is deterministic, so we need to perform that on batch_unguided
            means_if_unguided = {}
            for field in fields:
                _, mean_unguided, _ = noisers[field].forward_sde_step(
                    x=batch_unguided[field], t=t, dt=(t_hat - t)[0], batch_idx=batch.batch
                )
                means_if_unguided[field] = mean_unguided

            log_prob_xt_given_xtm1_pos = Normal(
                    loc=means_if_unguided['pos'],
                    scale=noisers['pos'].noise_weight * pos_diffusion * torch.sqrt(noise_dt.abs())
                ).log_prob(vals_hat['pos'])


            # compute the effective rotation matrices from means_if_unguided to vals_hat
            # each rotation is applied on the right (ie new_rot = old_rot * delta_rot, so vals_hat = means_if_unguided * equivalent_rotation_if_unguided)
            equivalent_rotation_if_unguided = torch.bmm(means_if_unguided['node_orientations'].mT, vals_hat['node_orientations'])

            log_prob_xt_given_xtm1_node_orientations = Normal(
                    loc=torch.zeros_like(diffusion_steps_hat['node_orientations']),
                    scale=(noisers['node_orientations'].noise_weight * node_orientations_diffusion * torch.sqrt(noise_dt.abs()))[:, None]
                ).log_prob(rotmat_to_rotvec(equivalent_rotation_if_unguided))

            # each of those log probabilities is of shape BN x 3
            # we now want to sum over the internal coordinates and compute
            # log probabilities of shape B--which is 1 for each particle

            assert log_prob_xt_given_xtm1_pos.ndim == 2
            assert log_prob_xt_given_xtm1_node_orientations.ndim == 2
            assert log_prob_xt_given_xtm1_pos.shape == log_prob_xt_given_xtm1_node_orientations.shape
            log_prob_xt_given_y_xtm1 = \
                    (log_prob_xt_given_y_xtm1_pos + log_prob_xt_given_y_xtm1_node_orientations).view(
                    batch_size,
                    -1,
                    3).sum(dim=(1,2))

            log_prob_xt_given_xtm1 = \
                    (log_prob_xt_given_xtm1_pos + log_prob_xt_given_xtm1_node_orientations).view(
                    batch_size,
                    -1,
                    3).sum(dim=(1,2))

            # sum values per particle
            # note: those are unnormalized weights
            log_w = log_prob_xt_given_xtm1 + guidance_prob - log_prob_xt_given_y_xtm1 - prior_guidance

            if not enable_guidance:
                assert torch.allclose(log_prob_xt_given_xtm1, log_prob_xt_given_y_xtm1, atol=1e-4)

            log_w += log_wtp1

            if i < 50:
                ess_threshold = 0.33
            elif i < 98:
                ess_threshold = 0.66
            elif i >= 98:
                ess_threshold = 1

            if i < resample_start:
                # don't resample yet, but act like we *could* resample so log_w gets accumulated
                resample_fn = resampling_function(ess_threshold=0.)
            else:
                # do resample
                resample_fn = resampling_function(ess_threshold=ess_threshold)

            if ess_threshold == 0: # never resample
                resample_indices = torch.arange(batch.num_graphs)
                log_wtp1 = log_w
            else:
                # resample
                resample_indices, is_resampled = resample_fn(log_w)

                if is_resampled:
                    print(f'resampled at step {i} selected particles {resample_indices}')

                resampled_pos = vals_hat['pos'].view(batch_size, -1, 3)[resample_indices]
                resampled_node_orientations = vals_hat['node_orientations'].view(batch_size, -1, 3, 3)[resample_indices]
                assert resampled_pos.shape[1] == resampled_node_orientations.shape[1]

                batch_hat = batch_hat.replace(pos=resampled_pos.reshape(-1, 3),
                                              node_orientations=resampled_node_orientations.reshape(-1, 3, 3))

                guidance_prob = guidance_prob[resample_indices]

                if is_resampled:
                    log_wtp1 = torch.zeros_like(log_w, device=device) # not accumulating

                    assert log_w.ndim == 1
                else:
                    # since we didn't resample because of the ess threshold, accumulate weights until we do
                    log_wtp1_normalized = normalize_log_weights(log_w, dim=0)
                    log_wtp1 = log_wtp1_normalized + np.log(batch.num_graphs)


        with torch.enable_grad():
            batch_hat['pos'] = batch_hat['pos'].requires_grad_()
            batch_hat['node_orientations'] = batch_hat['node_orientations'].requires_grad_()

            score_model.eval()

            score = _get_score(batch=batch_hat, t=t_hat, score_model=score_model, sdes=sdes)

            # First-order denoising step from t_hat to t_next.
            drift_hat = {}
            for field in fields:
                drift_hat[field], _ = predictors[field].reverse_drift_and_diffusion(
                    x=batch_hat[field], t=t_hat, batch_idx=batch.batch, score=score[field]
                )

            for field in fields:
                batch[field] = predictors[field].update_given_drift_and_diffusion(
                    x=batch_hat[field],
                    dt=(t_next - t_hat)[0],
                    drift=drift_hat[field],
                    diffusion=0.0,
                )[0]

        batch_unguided = {}
        # store this for the resampling step
        for field in fields:
            batch_unguided[field] = batch[field].detach()

        def log_p_y_given_x(x):
            dist = (torch.mean(x[:, c0 - 5:c0 + 5], axis=1) - torch.mean(x[:, c1 - 5:c1 + 5], axis=1)).square().sum(axis=-1).sqrt()
            # unnormalized probability. higher is more probable. 
            return -twist_k * (dist - beta)**2, dist

        def log_bias_potential_rmsd(x):
            # weights mask needs to be 1 x nresidues
            # we want to align on the same residues we're computing rmsd of
            x_aligned = weighted_rigid_align(x, untwisted_pos.unsqueeze(0), weights=ss_mask_region.unsqueeze(0))

            # Compute RMSD between aligned atom_pos and untwisted_pos
            mse_loss = ((x_aligned - untwisted_pos) ** 2).sum(dim=-1) # shape B x Natoms
            rmsd = torch.sqrt(
                torch.sum(mse_loss * ss_mask_region, dim=-1)
                / torch.sum(ss_mask_region, dim=-1)
            )
            if twist_k == 0:
                log_potential = torch.zeros_like(rmsd, device=device)
            else:
                log_potential = -twist_k * (rmsd - beta)**2 - np.log(np.sqrt(np.pi / twist_k))

            return log_potential, rmsd

        if (i > 30):
            guidance_performed = True

            prior_guidance = guidance_prob.clone() if isinstance(guidance_prob, torch.Tensor) else guidance_prob

            # Main loop modification
            with torch.enable_grad():
                if i < N-1:
                    final_estimate = first_order_denoiser_from_intermediate_ckpt(
                        sdes=sdes,
                        N=max((N - (i + 1)) // 10, 2),
                        eps_t=eps_t,
                        max_t=timesteps[i+1],
                        device=device,
                        batch=batch,
                        score_model=score_model,
                        noise=noise,
                        sequence=sequence,
                        **kwargs
                    )
                else:
                    final_estimate = batch

                # Compute guidance for the selected estimate
                if twist_rmsd:
                    guidance_prob, twist_dists = log_bias_potential_rmsd(final_estimate['pos'].view(batch_size, -1, 3))
                else:
                    guidance_prob, twist_dists = log_p_y_given_x(final_estimate['pos'].view(batch_size, -1, 3))

                guidance_sum = sum(guidance_prob) * extra_twist_k # apply extra twist that only gets used in gradients

            print(i, twist_dists)

            if enable_guidance and i > 30:

                # the sum is just so we have a scalar for autograd purposes;
                # it means we're taking d[p(y | particle 0) + p(y | particle 1) + ...]/dpos
                guidance_signal = {}
                guidance_signal['pos'], guidance_signal['node_orientations'] = \
                    torch.autograd.grad(guidance_sum,
                                            (batch_hat.pos,
                                            batch_hat.node_orientations))

                guidance_signal['pos'] = -1 * guidance_signal['pos']
                guidance_signal['node_orientations'] = -1 * guidance_signal['node_orientations']

                # now that we have guidance_signal, can detach
                guidance_prob = torch.tensor(guidance_prob).detach()
                twist_dists = torch.tensor(twist_dists).detach()

                ddr_final_ests.append(twist_dists)
                if twist_rmsd:
                    _, curr_dists = log_bias_potential_rmsd(batch['pos'].view(batch_size, -1, 3))
                else:
                    _, curr_dists = log_p_y_given_x(batch['pos'].view(batch_size, -1, 3))
                ddr_curr_ests.append(curr_dists)

                # take a first order step using the guidance score
                prior_pos = batch['pos']
                prior_orientations = batch['node_orientations']

                if i < 60:
                ## normal step for pos
                    batch['pos'] = predictors['pos'].update_given_drift_and_diffusion(
                        x=batch['pos'], # start from where the first order step ended (= batch_hat + dt score(batch_hat))
                        dt=(t_next - t_hat)[0],
                        drift=guidance_signal['pos'],
                        diffusion=0.0,
                    )[0]

                    ## update rotation matrix with exponential map
                    batch['node_orientations'], Rvec = update_rotation_matrix_from_gradient(
                            batch['node_orientations'],
                            guidance_signal['node_orientations'],
                            (t_next - t_hat)[0]
                    )
                    print('guided')

                if twist_rmsd:
                    _, curr_dists = log_bias_potential_rmsd(batch['pos'].view(batch_size, -1, 3))
                else:
                    _, curr_dists = log_p_y_given_x(batch['pos'].view(batch_size, -1, 3))
                ddr_post_guidance_ests.append(curr_dists)


        for field in fields:
            # whether or not we did guidance, detach so we don't track grads across iterations
            batch[field] = batch[field].detach()

    batch['twist_dist_final'] = torch.tensor(twist_dists)

    return batch



def heun_denoiser(
    *,
    sdes: dict[str, SDE],
    N: int,
    eps_t: float,
    max_t: float,
    device: torch.device,
    batch: Batch,
    score_model: torch.nn.Module,
    noise: float,
    sequence: str,
    **kwargs
) -> ChemGraph:
    """Sample from prior and then denoise."""

    batch = batch.to(device)
    if isinstance(score_model, torch.nn.Module):
        # permits unit-testing with dummy model
        score_model = score_model.to(device)
    assert isinstance(sdes["node_orientations"], torch.nn.Module)  # shut up mypy
    sdes["node_orientations"] = sdes["node_orientations"].to(device)
    batch = batch.replace(
        pos=sdes["pos"].prior_sampling(batch.pos.shape, device=device),
        node_orientations=sdes["node_orientations"].prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )

    ts_min = 0.0
    ts_max = 1.0
    timesteps = torch.linspace(max_t, eps_t, N, device=device)
    dt = -torch.tensor((max_t - eps_t) / (N - 1)).to(device)
    fields = list(sdes.keys())
    predictors = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=0.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }
    noisers = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=1.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }
    batch_size = batch.num_graphs

    for i in range(N):
        # Set the timestep
        t = torch.full((batch_size,), timesteps[i], device=device)
        t_next = t + dt  # dt is negative; t_next is slightly less noisy than t.

        # Select temporarily increased noise level t_hat.
        # To be more general than Algorithm 2 in Karras et al. we select a time step between the
        # current and the previous t.
        t_hat = t - noise * dt if (i > 0 and t[0] > ts_min and t[0] < ts_max) else t

        # Apply noise.
        vals_hat = {}
        for field in fields:
            vals_hat[field] = noisers[field].forward_sde_step(
                x=batch[field], t=t, dt=(t_hat - t)[0], batch_idx=batch.batch
            )[0]
        batch_hat = batch.replace(**vals_hat)

        score = _get_score(batch=batch_hat, t=t_hat, score_model=score_model, sdes=sdes)

        # First-order denoising step from t_hat to t_next.
        drift_hat = {}
        for field in fields:
            drift_hat[field], _ = predictors[field].reverse_drift_and_diffusion(
                x=batch_hat[field], t=t_hat, batch_idx=batch.batch, score=score[field]
            )

        for field in fields:
            batch[field] = predictors[field].update_given_drift_and_diffusion(
                x=batch_hat[field],
                dt=(t_next - t_hat)[0],
                drift=drift_hat[field],
                diffusion=0.0,
            )[0]

        # Apply 2nd order correction.
        if t_next[0] > 0.0:
            score = _get_score(batch=batch, t=t_next, score_model=score_model, sdes=sdes)

            drifts = {}
            avg_drift = {}
            for field in fields:
                drifts[field], _ = predictors[field].reverse_drift_and_diffusion(
                    x=batch[field], t=t_next, batch_idx=batch.batch, score=score[field]
                )

                avg_drift[field] = (drifts[field] + drift_hat[field]) / 2
            for field in fields:
                batch[field] = (
                    0.0
                    + predictors[field].update_given_drift_and_diffusion(
                        x=batch_hat[field],
                        dt=(t_next - t_hat)[0],
                        drift=avg_drift[field],
                        diffusion=0.0,
                    )[0]
                )

    return batch
