import torch  as th 
from torch.distributions import Normal 
import numpy as np 
import enum
from math import sqrt

from boltz.model.loss.diffusion import (
    weighted_rigid_align,
)

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

def safe_cat(list_of_tensors, dim=0):
    if len(list_of_tensors) == 0:
        return None 
    return th.cat(list_of_tensors, dim=dim)

def log_normal_density(sample, mean, var):
    return Normal(loc=mean, scale=th.sqrt(var)).log_prob(sample)

class TwistedDDPM:
    def __init__(self, 
                 sigmas,
                 gammas,
                 atom_mask,
                 noise_scale,
                 num_timesteps,
                 particle_base_shape, # should be (num_atoms, 3)
                 classifier_prob_fn, # bias function
                 denoise_fn, # (atom_coords, t_hat, multiplicity) -> atom_coords_denoised
                 num_diffn_samples, # diffusion_samples argument in main.py
                 device,
                 step_scale=1.0, # from Boltz code (1.638 used by default)
                 alignment_reverse_diff=True, # from Boltz code
                 ):

        self.T = num_timesteps
        self.particle_base_shape = particle_base_shape
        self.particle_base_dims = tuple(range(1, len(particle_base_shape)+1)) 
        self.num_diffn_samples = num_diffn_samples

        self.classifier_prob_fn = classifier_prob_fn
        self.denoise_fn = denoise_fn
        self.device = device
        self.step_scale = step_scale
        self.alignment_reverse_diff = alignment_reverse_diff

        self.use_mean_pred = True

        self.atom_mask = atom_mask
        self.shape = (*self.atom_mask.shape, 3)
        self.noise_scale = noise_scale

        # Convert sigmas and gammas to float numpy arras for consistency/accuracy
        sigmas = np.array(sigmas.cpu(), dtype=np.float32)
        sigmas = sigmas[::-1] ## flip sigma for the reverse time convention in TDS code
        self.sigmas = sigmas

        assert len(sigmas.shape) == 1, "sigmas must be 1-D"
        assert len(gammas) == len(sigmas), "gammas must match sigmas in size"

        gammas = np.array(gammas.cpu(), dtype=np.float32)
        gammas = gammas[::-1] ## flip gamma for the reverse time convention in TDS code
        self.gammas = gammas

        self.task = 'class_cond_gen' # this is fixed

        self.t_truncate = -1

        # model predicts x0
        self.model_mean_type = ModelMeanType.START_X

        self.model_var_type = ModelVarType.FIXED_SMALL

        self.clear_cache()

    def clear_cache(self):
        self.cache = {} 

    def ref_sample(self):
        # Sample initial positions using Boltz code 
        init_sigma = self.sigmas[-1]
        atom_coords = init_sigma * th.randn(self.shape, device=self.device)
        return atom_coords.requires_grad_()

    def ref_log_density(self, x):
        return Normal(loc=0, scale=1.0).log_prob(x) # from TDS code

    def p_mean_variance(
        self, atom_coords, atom_mask, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
    
        sigma_tm = self.sigmas[t+1].item()
        sigma_t = self.sigmas[t].item()
        gamma = self.gammas[t].item()

        t_hat = sigma_tm * (1 + gamma)
        
        std_t = self.noise_scale * sqrt(t_hat**2 - sigma_tm**2)
        noise = (
            std_t
            * th.randn(atom_coords.shape, device=self.device)
        )

        atom_coords_noisy = atom_coords + noise

        atom_coords_denoised = self.denoise_fn(
            atom_pos=atom_coords_noisy,
            t_hat=t_hat,
            multiplicity=atom_coords.shape[0], # allow batching
        )
        
        if self.alignment_reverse_diff:
            with th.autocast("cuda", enabled=False):
                try:
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )
                except:
                    import ipdb; ipdb.set_trace()

            atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

        s_k = (atom_coords_noisy - atom_coords_denoised) / t_hat # -1 * TDS score fn

        def process_xstart(x):
            if denoised_fn is not None: # set to None
                x = denoised_fn(x)
            if clip_denoised: # set to False
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(atom_coords_denoised)

        # Model mean as calculated in Boltz code
        # this is equivalent to x_t + sigma_t s_k^t tilde, where s_k^t tilde incorporates the addition of noise + denoising from that point
        # it's just a reordering of the TDS alg, in Boltz the noise is added before the denoising
        model_mean = atom_coords_noisy + self.step_scale * (sigma_t - t_hat) * s_k

        assert (
            model_mean.shape == pred_xstart.shape == atom_coords.shape
        )

        variance = std_t * th.ones_like(model_mean, device=self.device)
        log_variance = th.log(variance)

        return {
            "mean": model_mean,
            "variance": variance,
            "log_variance": log_variance,
            "pred_xstart": pred_xstart,
        } 

    def p_trans_model(self, xtp1, atom_mask, t, clip_denoised, model_kwargs):
        # compute mean and variance of p(x_t|x_{t+1}) under the model

        out = self.p_mean_variance(atom_coords=xtp1, atom_mask=atom_mask, t=t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        
        return {
            "mean_untwisted": out['mean'], 
            "var_untwisted": out['variance'], 
            "pred_xstart": out['pred_xstart']
        }

    def M(self, t, xtp1, extra_vals, P, device, collapse_proposal_shape=True):
        """ M: intial/transition distribution.
            x_T, extra_vals = M(T, None, extra_vals, P=P) # Initial sample x_T ~ M_0(dt)
            x_t, extra_vals = M(t, x_{t+1}, extra_vals) if t<T
            """
        if t == self.T:
            self.clear_cache()

            # line 2 in Alg 1 pseudocode
            xt = self.ref_sample(P)
            log_proposal = self.ref_log_density(xt)
            if collapse_proposal_shape:
                log_proposal = log_proposal.sum(self.particle_base_dims)

        else:
            # For t >= 1, sample from proposal p(x_t | x_{t+1}, y) 
            # For t = 0, sample from model p(x_0 | x_1)
            # the proposal distribution is precomputed in psi(x_{t+1}) from previous iteration

            resample_idcs = extra_vals[("resample_idcs", t+1)]

            xt_var = self.cache.pop(("xt_var", t))[resample_idcs] 
            xt_mean = self.cache.pop(("xt_mean", t))[resample_idcs]

            if self.use_mean_pred:
                xt = xt_mean
            else:
                # line 7 in Alg 1 pseudocode
                xt = xt_mean + th.randn_like(xt_mean) * th.sqrt(xt_var)
            assert not xt.requires_grad 

            if self.use_mean_pred and t == 0:
                log_proposal = 0
            else:
                log_proposal = log_normal_density(xt, xt_mean, xt_var)
                if collapse_proposal_shape:
                    log_proposal = log_proposal.sum(self.particle_base_dims)
           
        self.cache[('log_proposal', t)] = log_proposal
        return xt, extra_vals

    def G(self, t, xtp1, xt, extra_vals, debug_statistics=False, debug_info=None):
        """ G: potential function
            w_T, extra_vals = G(T, None, x_T, extra_vals) # Initial potential
            w_t, extra_vals = G(t, x_{t+1}, x_t, extra_vals) # subsequent potentials
            """
        P = xt.shape[0]
        assert xt.shape[1:] == self.particle_base_shape, xt.shape 

        log_proposal = self.cache.pop(('log_proposal', t))

        #######################################################
        # gathering previous and current log_potential values #
        #######################################################
        if t == self.T:
            log_potential_xtp1 = th.zeros_like(log_proposal)     
            log_p_trans_untwisted = log_proposal  

            if debug_statistics:
                self.debug_statistics = {'resample_idcs_tp1': [], 
                                        't': [],
                                        'log_target': [], 
                                        'log_p_trans_untwisted':[], 
                                        'log_potential_xt': [], 
                                        'log_potential_xtp1': [], 
                                        'log_proposal': [], 
                                        'log_w': []}

        else: 
            resample_idcs = extra_vals.pop(("resample_idcs", t+1))
            
            if debug_statistics:
                self.debug_statistics['resample_idcs_tp1'].append(resample_idcs.detach().cpu())
            
            log_potential_xtp1 = self.cache.pop(('log_potential', t+1)) # (P, C, H, W)
            log_potential_xtp1 = log_potential_xtp1[resample_idcs] 
            
            xt_mean_untwisted = self.cache.pop(('xt_mean_untwisted', t))[resample_idcs]
            xt_var = self.cache.pop(('xt_var_untwisted', t)) # all particles have the same variance

            # compute p_theta(x_t | x_{t+1}) under the model
            # Used in line 9 of Alg 1 pseudocode
            log_p_trans_untwisted = log_normal_density(xt, xt_mean_untwisted, xt_var)

            log_p_trans_untwisted = log_p_trans_untwisted.sum(dim=self.particle_base_dims)


        #######################################################
        # calculating log_potential_t and the proposal at t-1 #
        #######################################################

        if t > 0: 
            batch_p = extra_vals.get("batch_p", P) 
            xtm1_mean, log_potential_xt, mean_untwisted, var_untwisted, \
                pred_xstart, twisted_pred_xstart, grad_log_potential_xt = \
                self._compute_twisted_step(batch_p, xt, t, 
                                            model_kwargs=extra_vals.get('model_kwargs', None), 
                                            )
            logp_y_given_x0 = log_potential_xt
            
            self.cache[('xt_mean_untwisted', t-1)] = mean_untwisted
            self.cache[('xt_var_untwisted', t-1)] = var_untwisted
            if t == 1:
                # print("CACHE t=1", var_untwisted)
                self.cache[("xt_mean", t-1)] = mean_untwisted 
                self.cache[("xt_var", t-1)] = var_untwisted
            else:
                # print("CACHE t>1", var_untwisted)
                self.cache[("xt_mean", t-1)] = xtm1_mean
                self.cache[("xt_var", t-1)] = var_untwisted
            self.cache[('log_potential', t)] = log_potential_xt

            # calculate line 9 of Alg 1 pseudocode from the 4 quantities
            # log_p_trans_untwisted = p_theta(x^t | x^{t+1})
            # log_potential_xt = p_tilde^theta(y|x^t) 
            # log_potential_xtp1 = p_tilde^theta(y|x^{t+1}) from previous iteration
            # log_proposal = p_tilde_theta(x^t | x^{t+1}, y)
            log_target = log_p_trans_untwisted + log_potential_xt - log_potential_xtp1  
            log_w = log_target - log_proposal

            if self.t_truncate > 1 and t == self.t_truncate:
                pred_xstart = pred_xstart.to(xt.device)
                    
                self.xpred_at_t_truncate = pred_xstart
                 
                logp_y_given_x0 \
                    = self.classifier_prob_fn(xt=None, x0_hat=pred_xstart, atom_mask=self.atom_mask,
                                              return_grad=False) 
                self.log_w_x0_truncate = logp_y_given_x0 + log_p_trans_untwisted - log_potential_xtp1 - log_proposal 
                    
                pred_xstart = pred_xstart.cpu() 
                
        else:
            # t = 0 
            logp_y_given_x0 \
                = self.classifier_prob_fn(xt=None, x0_hat=xt, atom_mask=self.atom_mask, return_grad=False)
            
            grad_log_potential_xt = th.zeros_like(logp_y_given_x0) # for logging
            assert logp_y_given_x0.shape == (P,), logp_y_given_x0.shape 
            
            # suppose using mean prediction in the final step so proposal and diffusion target canceld out
            log_target =  logp_y_given_x0 - log_potential_xtp1  
            log_w = log_target

        return log_w, logp_y_given_x0, grad_log_potential_xt
    
    def _compute_twisted_step(self, batch_p, xt, t, model_kwargs):
        """compute xtm1_mean and xtm1_var given xt after applying the twisted operation""" 
        P = xt.shape[0]

        # Bring these in for scaling of the grad_log_potential_xt_batch
        # Call with t / t-1 instead of t+1 / t because we are 
        # calling p_trans_model with t = t-1
        sigma_tm = self.sigmas[t].item()
        sigma_t = self.sigmas[t-1].item()
        gamma = self.gammas[t-1].item()

        t_hat = sigma_tm * (1 + gamma)

        # split the batch into smaller batches
        xt_batches = th.split(xt.cpu(), batch_p)
        atom_mask_batches = th.split(self.atom_mask, batch_p)
        
        mean_untwisted = []
        var_untwisted = [] 
        pred_xstart = []  
        grad_log_potential_xt = [] 
        twisted_pred_xstart = [] 
        log_potential_xt = [] 
        xtm1_mean = []

        with th.enable_grad(), th.inference_mode(False):
            for xt_batch, atom_mask_batch in zip(xt_batches, atom_mask_batches):
                xt_batch = xt_batch.to(xt.device).requires_grad_() 
                out = self.p_trans_model(xtp1=xt_batch, atom_mask=atom_mask_batch, t=t-1, clip_denoised=False, 
                                        model_kwargs=model_kwargs)
                
                log_potential_xt_batch, grad_log_potential_xt_batch = self.classifier_prob_fn(xt=xt_batch, x0_hat=out['pred_xstart'], t=t, atom_mask=self.atom_mask)
                assert grad_log_potential_xt_batch.shape == xt_batch.shape
                assert log_potential_xt_batch.shape == (xt_batch.shape[0],)

                mean_untwisted.append(out['mean_untwisted'].detach())
                var_untwisted.append(out['var_untwisted'].detach())
                pred_xstart.append(out['pred_xstart'].detach().cpu()) 
                log_potential_xt.append(log_potential_xt_batch)

                xt_batch.requires_grad_(False)
                assert not log_potential_xt_batch.requires_grad 
                assert not grad_log_potential_xt_batch.requires_grad  

                # Mean computed for Boltz version of TDS Algorithm
                # Need to scale the grad_log_potential by sigma because it is really part of s_k
                grad_scale = self.step_scale * (sigma_t - t_hat) # this is a negative number
                xtm1_mean_batch = out['mean_untwisted'].detach() - grad_scale * grad_log_potential_xt_batch
                grad_log_potential_xt.append(grad_scale * grad_log_potential_xt_batch)
                xtm1_mean.append(xtm1_mean_batch) 

        return [safe_cat(lt) for lt in [xtm1_mean, log_potential_xt, mean_untwisted, var_untwisted, \
                                          pred_xstart, twisted_pred_xstart, grad_log_potential_xt]]

