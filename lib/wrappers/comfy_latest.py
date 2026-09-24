"""ComfyUI 1d48d9cf7bce ports (GPL-3.0); host access adapted for Forge.
Algorithm coefficients retained. See docs/UPSTREAM.md for provenance.
"""
from functools import partial
import torch
from scipy import integrate
from tqdm.auto import trange
from backend.patcher.base import set_model_options_post_cfg_function
from .comfy_ported import _model_sampling, _noise_sampler as default_noise_sampler
from .extra_samplers import sigma_to_half_log_snr, get_ancestral_step
from k_diffusion.sampling import to_d

@torch.no_grad()
def linear_multistep_coeff(order, t, i, j):
    if order - 1 > i:
        raise ValueError(f'Order {order} too high for step {i}')
    def fn(tau):
        prod = 1.
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod
    return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]

@torch.no_grad()
def _sample_cfgpp_history(model, x, sigmas, extra_args=None, callback=None, disable=None, history_weight=0.5, zero_weight=None, zero_order=1, uncond_history_weight=0.0):
    """CFG++ Euler with variable-step AB2 history and optional sigma-zero extrapolation."""
    extra_args = {} if extra_args is None else extra_args
    model_sampling = _model_sampling(model)
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    s_in = x.new_ones([x.shape[0]])
    sigmas_cpu = sigmas.detach().cpu().numpy()
    derivatives = []
    denoised_history = []
    old_uncond_d = None
    uncond_denoised = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"] if args["uncond"] is not None else args["cond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        alpha_s = sigmas[i] * lambda_fn(sigmas[i]).exp()
        alpha_t = sigmas[i + 1] * lambda_fn(sigmas[i + 1]).exp() if sigmas[i + 1] != 0 else sigmas[i + 1].new_ones([])
        current_uncond_d = to_d(x, sigmas[i], alpha_s * uncond_denoised)
        uncond_d = current_uncond_d
        dt = sigmas[i + 1] - sigmas[i]
        if i > 0 and uncond_history_weight:
            step_ratio = dt / (sigmas[i] - sigmas[i - 1])
            uncond_d = uncond_d + uncond_history_weight * step_ratio * (current_uncond_d - old_uncond_d)
        euler_step = alpha_t * denoised + sigmas[i + 1] * uncond_d - x
        d = euler_step / dt
        derivatives.append(d)
        if len(derivatives) > 2:
            derivatives.pop(0)

        if len(derivatives) == 1:
            step = euler_step
        else:
            coeffs = [linear_multistep_coeff(2, sigmas_cpu, i, j) for j in range(2)]
            history_step = sum(coeff * derivative for coeff, derivative in zip(coeffs, reversed(derivatives)))
            step = torch.lerp(euler_step, history_step, history_weight)
        x = x + step
        if sigmas[i + 1] == 0 and zero_weight is not None and denoised_history:
            if zero_order == 2 and len(denoised_history) > 1:
                sigma_0, sigma_1, sigma_2 = sigmas[i - 2], sigmas[i - 1], sigmas[i]
                weight_0 = sigma_1 * sigma_2 / ((sigma_0 - sigma_1) * (sigma_0 - sigma_2))
                weight_1 = sigma_0 * sigma_2 / ((sigma_1 - sigma_0) * (sigma_1 - sigma_2))
                weight_2 = sigma_0 * sigma_1 / ((sigma_2 - sigma_0) * (sigma_2 - sigma_1))
                zero_prediction = weight_0 * denoised_history[-2] + weight_1 * denoised_history[-1] + weight_2 * denoised
            else:
                denoised_slope = (denoised - denoised_history[-1]) / (sigmas[i] - sigmas[i - 1])
                zero_prediction = denoised - sigmas[i] * denoised_slope
            x = torch.lerp(x, zero_prediction, zero_weight)
        denoised_history.append(denoised)
        if len(denoised_history) > 2:
            denoised_history.pop(0)
        old_uncond_d = current_uncond_d
    return x

@torch.no_grad()
def sample_cfgpp_ud10_ab(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return _sample_cfgpp_history(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, history_weight=0.25, zero_weight=1.0, uncond_history_weight=0.1)

@torch.no_grad()
def sample_dpmpp_2s_ancestral_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_noise = s_noise * getattr(_model_sampling(model), "noise_scale", 1.0)

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

def sample_unipc_bh2(model, x, sigmas, extra_args=None, callback=None, disable=False):
    from modules.sd_samplers_extra import sample_unipc
    return sample_unipc(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, variant="bh2")
