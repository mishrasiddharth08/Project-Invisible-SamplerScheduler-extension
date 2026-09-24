from functools import partial

import torch
from tqdm.auto import trange
import torchsde
from scipy import integrate
from backend.patcher.base import set_model_options_post_cfg_function

from modules_forge.packages.k_diffusion import utils
from modules_forge.packages.k_diffusion.sampling import res_multistep

from . import sa_solver

def _is_const(sampling) -> bool:
    return sampling.prediction_type == "const"


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


class BrownianTreeNoiseSampler:
    """A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed, cpu=cpu)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


def sigma_to_half_log_snr(sigma, model_sampling):
    """Convert sigma to half-logSNR log(alpha_t / sigma_t)"""
    if _is_const(model_sampling):
        # log((1 - t) / t) = log((1 - sigma) / sigma)
        return sigma.logit().neg()
    return sigma.log().neg()

def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step"""
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(sigma_to, eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy"""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.cpu_tree = kwargs.pop("cpu", True)
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.pop("w0", None)
        if w0 is None:
            w0 = torch.zeros_like(x)
        self.batched = False
        if seed is None:
            seed = (torch.randint(0, 2**63 - 1, ()).item(),)
        elif isinstance(seed, (tuple, list)):
            if len(seed) != x.shape[0]:
                raise ValueError("Passing a list or tuple of seeds to BatchedBrownianTree requires a length matching the batch size.")
            self.batched = True
            w0 = w0[0]
        else:
            seed = (seed,)
        if self.cpu_tree:
            t0, w0, t1 = t0.detach().cpu(), w0.detach().cpu(), t1.detach().cpu()
        self.trees = tuple(torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed)

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        device, dtype = t0.device, t0.dtype
        if self.cpu_tree:
            t0, t1 = t0.detach().cpu().float(), t1.detach().cpu().float()
        w = torch.stack([tree(t0, t1) for tree in self.trees]).to(device=device, dtype=dtype) * (self.sign * sign)
        return w if self.batched else w[0]

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative"""
    return (x - denoised) / utils.append_dims(sigma, x.ndim)

def half_log_snr_to_sigma(half_log_snr, model_sampling):
    """Convert half-logSNR log(alpha_t / sigma_t) to sigma"""
    if _is_const(model_sampling):
        # 1 / (1 + exp(half_log_snr))
        return half_log_snr.neg().sigmoid()
    return half_log_snr.neg().exp()


def offset_first_sigma_for_snr(sigmas, model_sampling, percent_offset=1e-4):
    """Adjust the first sigma to avoid invalid logSNR"""
    if len(sigmas) <= 1:
        return sigmas
    if _is_const(model_sampling):
        if sigmas[0] >= 1:
            sigmas = sigmas.clone()
            sigmas[0] = model_sampling.percent_to_sigma(percent_offset)
    return sigmas

def ei_h_phi_1(h: torch.Tensor) -> torch.Tensor:
    """Compute the result of h*phi_1(h) in exponential integrator methods."""
    return torch.expm1(h)


def ei_h_phi_2(h: torch.Tensor) -> torch.Tensor:
    """Compute the result of h*phi_2(h) in exponential integrator methods."""
    return (torch.expm1(h) - h) / h

@torch.no_grad()
def sample_dpmpp_sde_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, eta=1.0, s_noise=1.0, r=0.5):
    """DPM-Solver++ (stochastic) with CFG++."""

    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]
    
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
    
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], temp[0])
            dt = sigmas[i + 1] - sigmas[i]
            x = denoised + d * sigmas[i + 1]
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * temp[0] + fac * temp[0]  # Use temp[0] instead of denoised
            x = denoised_2 + to_d(x, sigmas[i], denoised_d) * sd
            x = x + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su
    return x

@torch.no_grad()
def sample_dpmpp_2s_ancestral_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, eta=1.0, s_noise=1.0):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps and CFG++."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], temp[0])
            x = denoised + d * sigma_down
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            # r = torch.sinh(1 + (2 - eta) * (t_next - t) / (t - t_fn(sigma_up))) works only on non-cfgpp, weird
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_gradient_estimation(model, x, sigmas, extra_args=None, callback=None, disable=None, ge_gamma=2., cfg_pp=False):
    """Gradient-estimation sampler. Paper: https://openreview.net/pdf?id=o2ND9v0CeK"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    old_d = None

    uncond_denoised = None
    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    if cfg_pp:
        model_options = extra_args.get("model_options", {}).copy()
        extra_args["model_options"] = set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if cfg_pp:
            d = to_d(x, sigmas[i], uncond_denoised)
        else:
            d = to_d(x, sigmas[i], denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        dt = sigmas[i + 1] - sigmas[i]
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # Euler method
            if cfg_pp:
                x = denoised + d * sigmas[i + 1]
            else:
                x = x + d * dt

            if i >= 1:
                # Gradient estimation
                d_bar = (ge_gamma - 1) * (d - old_d)
                x = x + d_bar * dt
        old_d = d
    return x

@torch.no_grad()
def sample_gradient_estimation_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, ge_gamma=2.):
    return sample_gradient_estimation(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, ge_gamma=ge_gamma, cfg_pp=True)

@torch.no_grad()
def sample_res_multistep_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., noise_sampler=None):
    return res_multistep(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_noise=s_noise, noise_sampler=noise_sampler, eta=0., cfg_pp=True)

@torch.no_grad()
def sample_res_multistep_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    return res_multistep(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_noise=s_noise, noise_sampler=noise_sampler, eta=eta, cfg_pp=False)

@torch.no_grad()
def sample_res_multistep_ancestral_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    return res_multistep(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_noise=s_noise, noise_sampler=noise_sampler, eta=eta, cfg_pp=True)

@torch.no_grad()
def sample_seeds_2(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=0.5, solver_type="phi_1"):
    """SEEDS-2 - Stochastic Explicit Exponential Derivative-free Solvers (VP Data Prediction) stage 2.
    arXiv: https://arxiv.org/abs/2305.14267 (NeurIPS 2023)
    """
    if solver_type not in {"phi_1", "phi_2"}:
        raise ValueError("solver_type must be 'phi_1' or 'phi_2'")

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    inject_noise = eta > 0 and s_noise > 0

    model_sampling = model.inner_model.predictor
    sigma_fn = partial(half_log_snr_to_sigma, model_sampling=model_sampling)
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    fac = 1 / (2 * r)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigmas[i + 1] == 0:
            x = denoised
            continue

        lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
        h = lambda_t - lambda_s
        h_eta = h * (eta + 1)
        lambda_s_1 = torch.lerp(lambda_s, lambda_t, r)
        sigma_s_1 = sigma_fn(lambda_s_1)

        alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
        alpha_t = sigmas[i + 1] * lambda_t.exp()

        # Step 1
        x_2 = sigma_s_1 / sigmas[i] * (-r * h * eta).exp() * x - alpha_s_1 * ei_h_phi_1(-r * h_eta) * denoised
        if inject_noise:
            sde_noise = (-2 * r * h * eta).expm1().neg().sqrt() * noise_sampler(sigmas[i], sigma_s_1)
            x_2 = x_2 + sde_noise * sigma_s_1 * s_noise
        denoised_2 = model(x_2, sigma_s_1 * s_in, **extra_args)

        # Step 2
        if solver_type == "phi_1":
            denoised_d = torch.lerp(denoised, denoised_2, fac)
            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x - alpha_t * ei_h_phi_1(-h_eta) * denoised_d
        elif solver_type == "phi_2":
            b2 = ei_h_phi_2(-h_eta) / r
            b1 = ei_h_phi_1(-h_eta) - b2
            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x - alpha_t * (b1 * denoised + b2 * denoised_2)

        if inject_noise:
            segment_factor = (r - 1) * h * eta
            sde_noise = sde_noise * segment_factor.exp()
            sde_noise = sde_noise + segment_factor.mul(2).expm1().neg().sqrt() * noise_sampler(sigma_s_1, sigmas[i + 1])
            x = x + sde_noise * sigmas[i + 1] * s_noise
    return x


@torch.no_grad()
def sample_seeds_3(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r_1=1./3, r_2=2./3):
    """SEEDS-3 - Stochastic Explicit Exponential Derivative-free Solvers (VP Data Prediction) stage 3.
    arXiv: https://arxiv.org/abs/2305.14267 (NeurIPS 2023)
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    inject_noise = eta > 0 and s_noise > 0

    model_sampling = model.inner_model.predictor
    sigma_fn = partial(half_log_snr_to_sigma, model_sampling=model_sampling)
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigmas[i + 1] == 0:
            x = denoised
            continue

        lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
        h = lambda_t - lambda_s
        h_eta = h * (eta + 1)
        lambda_s_1 = torch.lerp(lambda_s, lambda_t, r_1)
        lambda_s_2 = torch.lerp(lambda_s, lambda_t, r_2)
        sigma_s_1, sigma_s_2 = sigma_fn(lambda_s_1), sigma_fn(lambda_s_2)

        alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
        alpha_s_2 = sigma_s_2 * lambda_s_2.exp()
        alpha_t = sigmas[i + 1] * lambda_t.exp()

        # Step 1
        x_2 = sigma_s_1 / sigmas[i] * (-r_1 * h * eta).exp() * x - alpha_s_1 * ei_h_phi_1(-r_1 * h_eta) * denoised
        if inject_noise:
            sde_noise = (-2 * r_1 * h * eta).expm1().neg().sqrt() * noise_sampler(sigmas[i], sigma_s_1)
            x_2 = x_2 + sde_noise * sigma_s_1 * s_noise
        denoised_2 = model(x_2, sigma_s_1 * s_in, **extra_args)

        # Step 2
        a3_2 = r_2 / r_1 * ei_h_phi_2(-r_2 * h_eta)
        a3_1 = ei_h_phi_1(-r_2 * h_eta) - a3_2
        x_3 = sigma_s_2 / sigmas[i] * (-r_2 * h * eta).exp() * x - alpha_s_2 * (a3_1 * denoised + a3_2 * denoised_2)
        if inject_noise:
            segment_factor = (r_1 - r_2) * h * eta
            sde_noise = sde_noise * segment_factor.exp()
            sde_noise = sde_noise + segment_factor.mul(2).expm1().neg().sqrt() * noise_sampler(sigma_s_1, sigma_s_2)
            x_3 = x_3 + sde_noise * sigma_s_2 * s_noise
        denoised_3 = model(x_3, sigma_s_2 * s_in, **extra_args)

        # Step 3
        b3 = ei_h_phi_2(-h_eta) / r_2
        b1 = ei_h_phi_1(-h_eta) - b3
        x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x - alpha_t * (b1 * denoised + b3 * denoised_3)
        if inject_noise:
            segment_factor = (r_2 - 1) * h * eta
            sde_noise = sde_noise * segment_factor.exp()
            sde_noise = sde_noise + segment_factor.mul(2).expm1().neg().sqrt() * noise_sampler(sigma_s_2, sigmas[i + 1])
            x = x + sde_noise * sigmas[i + 1] * s_noise
    return x

@torch.no_grad()
def sample_sa_solver(model, x, sigmas, extra_args=None, callback=None, disable=False, tau_func=None, s_noise=1.0, noise_sampler=None, predictor_order=3, corrector_order=4, use_pece=False, simple_order_2=False):
    """Stochastic Adams Solver with predictor-corrector method (NeurIPS 2023)."""
    if len(sigmas) <= 1:
        return x
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    model_sampling = model.inner_model.predictor
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
    lambdas = sigma_to_half_log_snr(sigmas, model_sampling=model_sampling)

    if tau_func is None:
        # Use default interval for stochastic sampling
        start_sigma = model_sampling.percent_to_sigma(0.2)
        end_sigma = model_sampling.percent_to_sigma(0.8)
        tau_func = sa_solver.get_tau_interval_func(start_sigma, end_sigma, eta=1.0)

    max_used_order = max(predictor_order, corrector_order)
    x_pred = x  # x: current state, x_pred: predicted next state

    h = 0.0
    tau_t = 0.0
    noise = 0.0
    pred_list = []

    # Lower order near the end to improve stability
    lower_order_to_end = sigmas[-1].item() == 0

    for i in trange(len(sigmas) - 1, disable=disable):
        # Evaluation
        denoised = model(x_pred, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x_pred, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        pred_list.append(denoised)
        pred_list = pred_list[-max_used_order:]

        predictor_order_used = min(predictor_order, len(pred_list))
        if i == 0 or (sigmas[i + 1] == 0 and not use_pece):
            corrector_order_used = 0
        else:
            corrector_order_used = min(corrector_order, len(pred_list))

        if lower_order_to_end:
            predictor_order_used = min(predictor_order_used, len(sigmas) - 2 - i)
            corrector_order_used = min(corrector_order_used, len(sigmas) - 1 - i)

        # Corrector
        if corrector_order_used == 0:
            # Update by the predicted state
            x = x_pred
        else:
            curr_lambdas = lambdas[i - corrector_order_used + 1:i + 1]
            b_coeffs = sa_solver.compute_stochastic_adams_b_coeffs(
                sigmas[i],
                curr_lambdas,
                lambdas[i - 1],
                lambdas[i],
                tau_t,
                simple_order_2,
                is_corrector_step=True,
            )
            pred_mat = torch.stack(pred_list[-corrector_order_used:], dim=1)    # (B, K, ...)
            corr_res = torch.tensordot(pred_mat, b_coeffs, dims=([1], [0]))  # (B, ...)
            x = sigmas[i] / sigmas[i - 1] * (-(tau_t ** 2) * h).exp() * x + corr_res

            if tau_t > 0 and s_noise > 0:
                # The noise from the previous predictor step
                x = x + noise

            if use_pece:
                # Evaluate the corrected state
                denoised = model(x, sigmas[i] * s_in, **extra_args)
                pred_list[-1] = denoised

        # Predictor
        if sigmas[i + 1] == 0:
            # Denoising step
            x_pred = denoised
        else:
            tau_t = tau_func(sigmas[i + 1])
            curr_lambdas = lambdas[i - predictor_order_used + 1:i + 1]
            b_coeffs = sa_solver.compute_stochastic_adams_b_coeffs(
                sigmas[i + 1],
                curr_lambdas,
                lambdas[i],
                lambdas[i + 1],
                tau_t,
                simple_order_2,
                is_corrector_step=False,
            )
            pred_mat = torch.stack(pred_list[-predictor_order_used:], dim=1)    # (B, K, ...)
            pred_res = torch.tensordot(pred_mat, b_coeffs, dims=([1], [0]))  # (B, ...)
            h = lambdas[i + 1] - lambdas[i]
            x_pred = sigmas[i + 1] / sigmas[i] * (-(tau_t ** 2) * h).exp() * x + pred_res

            if tau_t > 0 and s_noise > 0:
                noise = noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * tau_t ** 2 * h).expm1().neg().sqrt() * s_noise
                x_pred = x_pred + noise
    return x_pred


@torch.no_grad()
def sample_sa_solver_pece(model, x, sigmas, extra_args=None, callback=None, disable=False, tau_func=None, s_noise=1.0, noise_sampler=None, predictor_order=3, corrector_order=4, simple_order_2=False):
    """Stochastic Adams Solver with PECE (Predict–Evaluate–Correct–Evaluate) mode (NeurIPS 2023)."""
    return sample_sa_solver(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, tau_func=tau_func, s_noise=s_noise, noise_sampler=noise_sampler, predictor_order=predictor_order, corrector_order=corrector_order, use_pece=True, simple_order_2=simple_order_2)

@torch.no_grad()
def sample_exp_heun_2_x0(model, x, sigmas, extra_args=None, callback=None, disable=None, solver_type="phi_2"):
    """Deterministic exponential Heun second order method in data prediction (x0) and logSNR time."""
    return sample_seeds_2(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=0.0, s_noise=0.0, noise_sampler=None, r=1.0, solver_type=solver_type)


@torch.no_grad()
def sample_exp_heun_2_x0_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type="phi_2"):
    """Stochastic exponential Heun second order method in data prediction (x0) and logSNR time."""
    return sample_seeds_2(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler, r=1.0, solver_type=solver_type)

@torch.no_grad()
def sample_euler_a2(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, eta=1.0, s_noise=1.0, extrapolation=0.425):
    """Euler ancestral sampler that averages two noise paths and extrapolates along their mean direction."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigmas[i + 1] == 0:
            x = denoised
            continue

        downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
        sigma_down = sigmas[i + 1] * downstep_ratio
        alpha_ip1 = 1 - sigmas[i + 1]
        alpha_down = 1 - sigma_down

        sigma_down_i_ratio = sigma_down / sigmas[i]
        deterministic_path = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised

        if eta > 0 and s_noise != 0:
            base = (alpha_ip1 / alpha_down) * deterministic_path
            renoise_coeff = (sigmas[i + 1]**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2).clamp_min(0).sqrt()
            noise_scale = s_noise * renoise_coeff

            noise_1 = noise_sampler(sigmas[i], sigmas[i + 1])
            noise_2 = noise_sampler(sigmas[i], sigmas[i + 1])

            path_1 = base + noise_1 * noise_scale
            path_2 = base + noise_2 * noise_scale
            merged = 0.5 * (path_1 + path_2)
            direction = merged - base
            x = merged + extrapolation * direction
        else:
            x = deterministic_path
    return x

# Registration lives in lib/register.py - this module only defines sampler math.
