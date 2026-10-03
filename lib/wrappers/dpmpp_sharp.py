"""DPM++ 2M Sharp family, ported from envy-ai/ComfyUI-DPMpp-2M-Sharp.

Upstream ships these as a ComfyUI V3 custom node. This port keeps the sampler
maths verbatim and swaps the three host-API touch points for the helpers this
extension already uses elsewhere:

* ``comfy.k_diffusion.sampling`` lookups -> ``.extra_samplers``
* ``comfy.model_sampling.CONST`` isinstance check -> ``comfy_ported._is_rectified_flow``
* ``model_patcher.get_model_object("model_sampling")`` -> ``comfy_ported._model_sampling``

The RES 2S/2M ``_nc`` paths are upstream's own bundled extraction from RES4LYF
(commit 5e72fe3); see LICENSE.RES4LYF carried in ``licenses/`` provenance notes.
Upstream licence: see NOTICE and the upstream repository.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from functools import partial

import torch
from tqdm.auto import trange

from .comfy_ported import _is_rectified_flow, _model_sampling, _noise_sampler
from .extra_samplers import (
    BrownianTreeNoiseSampler,
    default_noise_sampler,
    ei_h_phi_1,
    ei_h_phi_2,
    half_log_snr_to_sigma,
    offset_first_sigma_for_snr,
    sigma_to_half_log_snr,
)


# --------------------------------------------------------------------------
# DPM++ 2M Sharp / DPM++ 2M SDE GPU Sharp / SEEDS_2 Sharp
# --------------------------------------------------------------------------
def sample_dpmpp_2m_sharp(model, x, sigmas, extra_args=None, callback=None, disable=None, sharpness=0.15):
    """DPM-Solver++(2M) with progressively sharpened denoised history."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        sigma_progress = i / len(sigmas)
        adjustment_factor = 1 + sharpness * sigma_progress * sigma_progress
        old_denoised = denoised * adjustment_factor
    return x


def sample_dpmpp_2m_sde_gpu_sharp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None, solver_type="midpoint", sharpness=0.15):
    """DPM-Solver++(2M) SDE with GPU noise and sharpened denoised history."""
    if len(sigmas) <= 1:
        return x

    if solver_type not in {"heun", "midpoint"}:
        raise ValueError("solver_type must be 'heun' or 'midpoint'")

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = (
        BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=False)
        if noise_sampler is None
        else noise_sampler
    )
    s_in = x.new_ones([x.shape[0]])

    model_sampling = _model_sampling(model)
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
    s_noise = s_noise * getattr(model_sampling, "noise_scale", 1.0)

    old_denoised = None
    h, h_last = None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)
            alpha_t = sigmas[i + 1] * lambda_t.exp()

            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == "heun":
                    x = x + alpha_t * ((-h_eta).expm1().neg() / (-h_eta) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == "midpoint":
                    x = x + 0.5 * alpha_t * (-h_eta).expm1().neg() * (1 / r) * (denoised - old_denoised)

            if eta > 0 and s_noise > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        sigma_progress = i / len(sigmas)
        adjustment_factor = 1 + sharpness * sigma_progress * sigma_progress
        old_denoised = denoised * adjustment_factor
        h_last = h
    return x


def sample_seeds_2_sharp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None, r=0.5, solver_type="phi_1", sharpness=0.15):
    """SEEDS-2 with progressively scaled first-stage prediction differences."""
    if solver_type not in {"phi_1", "phi_2"}:
        raise ValueError("solver_type must be 'phi_1' or 'phi_2'")

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    # Forge's default noise sampler takes no seed; keep seeded reproducibility
    # the same way comfy_ported does for its own SDE ports.
    if noise_sampler is None:
        noise_sampler = _noise_sampler(x, seed) if seed is not None else default_noise_sampler(x)
    s_in = x.new_ones([x.shape[0]])

    model_sampling = _model_sampling(model)
    s_noise = s_noise * getattr(model_sampling, "noise_scale", 1.0)
    inject_noise = eta > 0 and s_noise > 0
    sigma_fn = partial(half_log_snr_to_sigma, model_sampling=model_sampling)
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    fac = 1 / (2 * r)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

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

        x_2 = sigma_s_1 / sigmas[i] * (-r * h * eta).exp() * x - alpha_s_1 * ei_h_phi_1(-r * h_eta) * denoised
        if inject_noise:
            sde_noise = (-2 * r * h * eta).expm1().neg().sqrt() * noise_sampler(sigmas[i], sigma_s_1)
            x_2 = x_2 + sde_noise * sigma_s_1 * s_noise
        denoised_2 = model(x_2, sigma_s_1 * s_in, **extra_args)

        adjustment_factor = 1 + sharpness * (i / len(sigmas)) ** 2
        if solver_type == "phi_1":
            denoised_d = torch.lerp(denoised, denoised_2, fac)
            if sharpness != 0.0:
                denoised_d = denoised_d - fac * (adjustment_factor - 1) * denoised
            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x - alpha_t * ei_h_phi_1(-h_eta) * denoised_d
        elif solver_type == "phi_2":
            b2 = ei_h_phi_2(-h_eta) / r
            b1 = ei_h_phi_1(-h_eta) - b2
            denoised_d = b1 * denoised + b2 * denoised_2
            if sharpness != 0.0:
                denoised_d = denoised_d - b2 * (adjustment_factor - 1) * denoised
            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x - alpha_t * denoised_d

        if inject_noise:
            segment_factor = (r - 1) * h * eta
            sde_noise = sde_noise * segment_factor.exp()
            sde_noise = sde_noise + segment_factor.mul(2).expm1().neg().sqrt() * noise_sampler(sigma_s_1, sigmas[i + 1])
            x = x + sde_noise * sigmas[i + 1] * s_noise
    return x


# --------------------------------------------------------------------------
# RES 2S/2M (no-correction) paths, upstream's RES4LYF extraction
# --------------------------------------------------------------------------
def _phi(order, h, c=1.0):
    # Match RES4LYF's high-precision analytic coefficients without mpmath.
    with localcontext() as context:
        context.prec = 80
        z = -Decimal.from_float(float(h)) * Decimal.from_float(c)
        if z == 0:
            return 1.0 if order == 1 else 0.5
        remainder = z.exp() - 1
        if order == 2:
            remainder -= z
        return float(remainder / z ** order)


def _reanchor(x_0, x, data, sigma, sub_sigma):
    anchored = (x_0 - data) / sigma
    unmoored = (x - data) / sub_sigma
    return x_0 - sigma * (unmoored + (anchored - unmoored))


def _rebound(x_0, x, sigma, sigma_next):
    eps = (x_0 - x) / (sigma - sigma_next)
    return x_0 - sigma * eps + sigma_next * eps


def _swap_noise(x_0, x, sigma, sigma_next, generator, variance_preserving):
    sigma_up = sigma_next * 0.5
    residual = (sigma_next ** 2 - sigma_up ** 2) ** 0.5
    alpha = 1 - sigma_next + residual if variance_preserving else torch.ones_like(sigma_next)
    sigma_down = residual / alpha
    eps = (x_0 - x) / (sigma - sigma_next)
    data = x_0 - sigma * eps
    noise = torch.randn(x.shape, dtype=torch.float64, layout=x.layout, device=x.device, generator=generator)
    noise = (noise - noise.mean()) / noise.std()
    noise.sub_(noise.mean(dim=(-2, -1), keepdim=True)).div_(noise.std(dim=(-2, -1), keepdim=True))
    return alpha * (data + sigma_down * eps) + sigma_up * noise


def _model_sampling_bounds(model_sampling, x, sigmas):
    """sigma_min / sigma_max for the host predictor, with a sigma fallback.

    ComfyUI's model_sampling objects carry these; Forge's predictor may not,
    in which case the schedule itself is a safe source for the same bounds.
    """
    sigma_min = getattr(model_sampling, "sigma_min", None)
    sigma_max = getattr(model_sampling, "sigma_max", None)
    if sigma_min is None or sigma_max is None:
        positive = sigmas[sigmas > 0]
        sigma_min = positive.min() if len(positive) else sigmas.new_tensor(0.0)
        sigma_max = sigmas.max()
    return (
        sigma_min.to(device=x.device, dtype=torch.float64),
        sigma_max.to(device=x.device, dtype=torch.float64),
    )


def _sample_res(model, x, sigmas, extra_args, callback, disable, multistep, sharpness):
    if len(sigmas) <= 1:
        return x

    model_sampling = _model_sampling(model)
    variance_preserving = _is_rectified_flow(model)
    sigma_min, sigma_max = _model_sampling_bounds(model_sampling, x, sigmas)
    sigmas = sigmas.to(device=x.device, dtype=torch.float64).clone()
    sample_sigmas = sigmas
    sigmas = torch.unique_consecutive(sigmas)
    unsample_from_zero = bool(sigmas[0] == 0 and sigmas[-1] == 0)
    if sigmas[0] == 0:
        sigmas = sigmas[1:]
        if len(sigmas) and sigmas[-1] == 0:
            sigmas = sigmas[:-1]
    if len(sigmas) <= 1:
        return x
    if sigmas[-1] == 0:
        if sigmas[-2] < sigma_min:
            sigmas[-2] = sigma_min
        elif (sigmas[-2] - sigma_min).abs() > 1e-4:
            sigmas = torch.cat((sigmas[:-1], sigma_min.unsqueeze(0), sigmas[-1:]))
    elif unsample_from_zero and not torch.isclose(sigmas[0], sigma_min):
        sigmas = torch.cat((sigma_min.unsqueeze(0), sigmas))
    steps = len(sigmas) - (2 if sigmas[-1] == 0 else 1)

    extra_args = {} if extra_args is None else extra_args.copy()
    options = extra_args["model_options"] = extra_args.get("model_options", {}).copy()
    transformer_options = options["transformer_options"] = options.get("transformer_options", {}).copy()
    transformer_options["sample_sigmas"] = sample_sigmas
    noise = torch.Generator(device=x.device).manual_seed(torch.initial_seed() + 1)
    substep_noise = torch.Generator(device=x.device).manual_seed(9999)
    x = x.to(torch.float32)
    s_in = x.new_ones([x.shape[0]])
    history = None
    stages = torch.zeros((3, *x.shape), dtype=x.dtype, device=x.device)
    eps = torch.zeros((2, *x.shape), dtype=x.dtype, device=x.device)
    data = torch.zeros_like(eps)

    for step in trange(steps, disable=disable):
        sigma, sigma_next = sigmas[step:step + 2]
        h = -(sigma_next / sigma).log()
        use_history = multistep and step >= 2 and bool(h < 1)
        euler = multistep and bool(h >= 1) and bool(sigma < 0.1)
        c2 = float((sigmas[step] / sigmas[step - 1]).log() / h) if use_history else 0.5
        if euler:
            a = torch.zeros(1, device=x.device, dtype=eps.dtype)
            b = torch.ones(1, device=x.device, dtype=eps.dtype)
            stage_sigmas = (-(-sigma.log() + h * sigmas.new_tensor([0, 1]))).exp()
        else:
            a = torch.tensor([c2 * _phi(1, h, c2), 0], device=x.device, dtype=eps.dtype)
            b2 = _phi(2, h) / c2
            b = torch.tensor([_phi(1, h) - b2, b2], device=x.device, dtype=eps.dtype)
            stage_sigmas = (-(-sigma.log() + h * sigmas.new_tensor([0, c2, 1]))).exp()
        zero_coefficients = torch.zeros_like(a)
        rows = 1 if use_history or euler else 2
        stages[0] = x
        x_0 = stages[0].clone()
        if use_history:
            eps[1] = history - x_0

        for row in range(rows):
            sub_sigma = stage_sigmas[row]
            transformer_options.update(row=row, x_tmp=stages[row], sigma_next=sigma_next)
            prediction = model(stages[row], sub_sigma * s_in, **extra_args)
            prediction = _reanchor(x_0, stages[row], prediction, sigma, sub_sigma)
            eps[row] = prediction - x_0
            data[row] = prediction

            final_stage = row == rows - 1
            sub_sigma_next = stage_sigmas[-1] if final_stage else stage_sigmas[1]
            sub_h = -(sub_sigma_next / sigma).log()
            h_new = h * sub_h / sub_h if sub_h != 0 else h
            eps_update = eps
            if not multistep and final_stage and sharpness != 0:
                factor = 1 + sharpness * (step / len(sigmas)) ** 2
                eps_update = eps.clone()
                eps_update[0] += (data[0] * factor - x_0) - (data[0] - x_0)
            coefficients = b if final_stage else a
            stages[row + 1] = x_0 + h_new * torch.einsum("i, i... -> ...", coefficients, eps_update[: len(coefficients)])
            if sigma > sub_sigma_next:
                stages[row + 1] = _rebound(x_0, stages[row + 1], sigma, sub_sigma_next)
            if not final_stage:
                stages[row + 1] = _swap_noise(x_0, stages[row + 1], sigma, sub_sigma_next, substep_noise, variance_preserving)
                # RES4LYF reanchors the noisy intermediate predictor before stage two.
                if stage_sigmas[row] > sigma_min and h < sigma_max / 2 and sigma > 0.03:
                    for _ in range(100):
                        x_0 = stages[1] - h * torch.einsum("i, i... -> ...", a, eps)
                        stages[0] = x_0 + h * torch.einsum("i, i... -> ...", zero_coefficients, eps)
                        eps[0] = _reanchor(x_0, stages[0], data[0], sigma, stage_sigmas[0]) - x_0

        x = _rebound(x_0, stages[rows], sigma, sigma_next)
        x = _swap_noise(x_0, x, sigma, sigma_next, noise, variance_preserving)
        if callback is not None:
            callback({"x": x, "i": step, "i_sched": step, "final": False, "sigma": sigma, "sigma_next": sigma_next, "denoised": data[0]})
        history = data[0].clone()
        if multistep and sharpness != 0:
            history *= 1 + sharpness * (step / len(sigmas)) ** 2

    if steps and callback is not None:
        callback({"x": x, "i": steps, "i_sched": steps, "final": True, "sigma": sigma, "sigma_next": sigma_next, "denoised": data[0]})
    return x


def sample_res_2s_nc(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return _sample_res(model, x, sigmas, extra_args, callback, disable, False, 0.0)


def sample_res_2m_nc(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return _sample_res(model, x, sigmas, extra_args, callback, disable, True, 0.0)


def sample_res_2s_nc_sharp(model, x, sigmas, extra_args=None, callback=None, disable=None, sharpness=0.15):
    return _sample_res(model, x, sigmas, extra_args, callback, disable, False, sharpness)


def sample_res_2m_nc_sharp(model, x, sigmas, extra_args=None, callback=None, disable=None, sharpness=0.15):
    return _sample_res(model, x, sigmas, extra_args, callback, disable, True, sharpness)