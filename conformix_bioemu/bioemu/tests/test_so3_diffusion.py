# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import torch
from scipy.stats import wasserstein_distance

from bioemu.so3_sde import (
    DiGSO3SDE,
    ScoreSO3,
    rotmat_to_rotvec,
    skew_matrix_exponential_map_axis_angle,
    vector_to_skew_matrix,
)


# Exponential map from so(3) to SO(3), this is the matrix exponential
def exp(A):
    rotvec = rotmat_to_rotvec(A)
    angles = torch.norm(rotvec, dim=-1)
    vectors = rotvec / (angles[..., None] + 1e-5)

    # Exponentials using Rodrigues' formula.
    skewmats = -vector_to_skew_matrix(vectors)
    exp_map = skew_matrix_exponential_map_axis_angle(angles, skewmats)
    return exp_map


# Exponential map from tangent space at R0 to SO(3)
def expmap(R0, tangent):
    skew_sym = torch.einsum("...ij,...ik->...jk", R0, tangent)
    return torch.einsum("...ij,...jk->...ik", R0, exp(skew_sym))


# Normal sample in tangent space at R0
def tangent_gaussian(R0):
    return torch.einsum(
        "...ij,...jk->...ik", R0, vector_to_skew_matrix(torch.randn(R0.shape[0], 3))
    )


# Simulates a Geodesic Random Walk (GRW) on a Riemannian manifold.
# Implements Algorithm 1 in https://arxiv.org/pdf/2202.02763.pdf
def geodesic_random_walk(p_initial, drift, diffusion, ts):
    Rts = {ts[0]: p_initial()}
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]  # negative for reverse process
        Rts[ts[i]] = expmap(
            Rts[ts[i - 1]],
            dt * drift(Rts[ts[i - 1]], ts[i - 1])  # Drift
            + diffusion(ts[i - 1])
            * np.sqrt(abs(dt))
            * tangent_gaussian(Rts[ts[i - 1]]),  # Diffusion
        )
    return Rts


def test_so3_score(
    num_samples: int = 1000,
    t_truncate: float = 0.1,
    dist_threshold: float = 0.2,
):
    """
    Unit test for ensuring forward SDE matches reverse SDE.
    Following https://arxiv.org/pdf/2202.02763.pdf we simulate
    a geodesic random walk (GRW) from algorithm 1 of the forward
    and reverse SDE on manifolds which are given as equations
    (3) and (4) respectively. This test effectively sees if the
    score is able to simulate the reverse process correctly.

    We test if the Wasserstein distance between the distributions of
    every Euler Maruyama discretization step is below a threshold (dist_threshold).

    Args:
        num_samples: Number of rotations to simulate.
            Higher will lead to better distribution estimation but more compute.
        t_truncate: value of t to truncate when comparing distributions.
            Small t can be instable due to the truncated IGSO(3) density and number of samples
        dist_threshold: threshold for the Wasserstein distance of when we consider two distributions
            to be the same.
        timeout: allowed time (in seconds) for the test to run.
    """
    np.random.seed(seed=1)
    torch.manual_seed(1)

    # Set-up SDE
    # We use sigma_max=2.5 to make sure the SDE converges to the uniform density.
    # 1.5 is used in FrameDiff because converging is not necessary.
    sde = DiGSO3SDE(
        eps_t=1e-4,
        sigma_min=0.1,
        sigma_max=2.5,
        l_max=1000,
    )

    # Discretization of [0, T]
    ts = np.linspace(0, sde.T, 200)

    # Simulate forward GRW
    forward_samples = geodesic_random_walk(
        # Surrogate data distribution mocked as identity rotations.
        p_initial=lambda: torch.eye(3)[None].repeat(num_samples, 1, 1),
        drift=lambda Rt, t: 0.0,
        diffusion=lambda t: sde.beta(torch.tensor(t)).type(torch.float64),
        ts=ts,
    )

    igso3_score = ScoreSO3(num_omega=1000, sigma_grid=torch.linspace(0.1, 2.5, 1000))

    def sampling_score_t(Rt, t):
        sigma = sde._marginal_std(torch.tensor(t))
        grad_coefficients = igso3_score.forward(
            sigma * torch.ones(Rt.shape[0]), rotmat_to_rotvec(Rt)
        )
        return torch.einsum("...ij,...jk->...ik", Rt, vector_to_skew_matrix(grad_coefficients))

    # Simulate reverse GRW
    reverse_samples = geodesic_random_walk(
        p_initial=lambda: sde.prior_sampling((num_samples,)),  # Uniform over SO(3).
        drift=lambda Rt, t: -sde.beta(torch.tensor(t)).type(torch.float64) ** 2
        * sampling_score_t(Rt, t),
        diffusion=lambda t: sde.beta(torch.tensor(t)).type(torch.float64),
        ts=ts[::-1],
    )
    for t in ts:
        # Skip calculating distances for t less than a cutoff.
        if t < t_truncate:
            continue
        Rt_forward = rotmat_to_rotvec(forward_samples[t])
        Rt_reverse = rotmat_to_rotvec(reverse_samples[t])

        omega_forward = torch.norm(Rt_forward, dim=-1).numpy()
        omega_reverse = torch.norm(Rt_reverse, dim=-1).numpy()

        dist = wasserstein_distance(omega_forward, omega_reverse)
        assert dist < dist_threshold, f"Wasserstein distance too large for t={t}."
