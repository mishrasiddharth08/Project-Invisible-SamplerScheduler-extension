"""Scheduler (sigma-schedule) implementations merged into this engine.

Sources, all already proven against Forge Neo before being merged here:

* ``cosine`` / ``cosexp`` / ``phi`` / ``laplace`` / ``karras_dyn`` / ``custom``
  - from the ``webUI_ExtraSchedulers`` extension.
* ``tan`` / ``beta57`` - from the ``sd-forge-res4lyf`` extension, matching
  ClownsharkBatwing/RES4LYF and cyberdeliaAI/forge-beta57-scheduler.

Signature contract (set by ``modules/sd_samplers_kdiffusion.py``): Forge calls
``scheduler.function(n=steps, sigma_min=..., sigma_max=..., device=...)`` and
additionally passes ``inner_model=`` when ``Scheduler.need_inner_model`` is
True.  Every function below returns ``n + 1`` sigmas ending in ``0.0``, which is
what the sampling loop expects.
"""

from __future__ import annotations

import math

import numpy
import torch

# Anything the ``custom`` expression is allowed to touch. Deliberately tiny:
# the expression is evaluated, so it gets no builtins and no module access.
_SAFE_EVAL_GLOBALS = {
    "__builtins__": {},
    "abs": abs,
    "min": min,
    "max": max,
    "pow": pow,
    "round": round,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "atan": math.atan,
    "floor": math.floor,
    "ceil": math.ceil,
}

#: Fallback expression, mirrored into ``config.json`` and the Settings entry.
DEFAULT_CUSTOM_SIGMAS = "m + (M-m)*(1-x)**3"


def _custom_expression() -> str:
    """Read the custom sigma expression without requiring any generation UI.

    Priority: Forge Settings entry -> ``config.json`` -> built-in default.
    """
    try:
        from modules import shared

        value = getattr(shared.opts, "pi_ss_custom_sigmas", None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    except Exception:
        pass
    try:
        from .. import CONFIG_PATH, read_json

        cfg = read_json(CONFIG_PATH, {}) or {}
        value = cfg.get("custom_sigmas")
        if isinstance(value, str) and value.strip():
            return value.strip()
    except Exception:
        pass
    return DEFAULT_CUSTOM_SIGMAS


# --------------------------------------------------------------------------
# webUI_ExtraSchedulers family
# --------------------------------------------------------------------------
def cosine_scheduler(n, sigma_min, sigma_max, device):
    sigmas = torch.zeros(n, device=device)
    if n == 1:
        sigmas[0] = sigma_max
    else:
        for x in range(n):
            p = x / (n - 1)
            sigmas[x] = sigma_min + 0.5 * (sigma_max - sigma_min) * (1 - math.cos(math.pi * (1 - p**0.5)))
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def cosexpblend_scheduler(n, sigma_min, sigma_max, device):
    sigmas = []
    if n == 1:
        sigmas.append(sigma_max)
    else:
        K = (sigma_min / sigma_max) ** (1 / (n - 1))
        E = sigma_max
        for x in range(n):
            p = x / (n - 1)
            C = sigma_min + 0.5 * (sigma_max - sigma_min) * (1 - math.cos(math.pi * (1 - p**0.5)))
            sigmas.append(C + p * (E - C))
            E *= K
    sigmas += [0.0]
    return torch.FloatTensor(sigmas).to(device)


def phi_scheduler(n, sigma_min, sigma_max, device):
    """Phi scheduler, modified from the original by @extraltodeus."""
    sigmas = torch.zeros(n, device=device)
    if n == 1:
        sigmas[0] = sigma_max
    else:
        phi = (1 + 5**0.5) / 2
        for x in range(n):
            sigmas[x] = sigma_min + (sigma_max - sigma_min) * ((1 - x / (n - 1)) ** (phi * phi))
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def laplace_scheduler(n, sigma_min, sigma_max, device="cpu"):
    """Noise schedule proposed by Tiankai et al. (2024)."""
    mu = 0.0
    beta = 0.5
    epsilon = 1e-5  # avoid log(0)
    x = torch.linspace(0, 1, n, device=device)
    lmb = mu - beta * torch.sign(0.5 - x) * torch.log(1 - 2 * torch.abs(0.5 - x) + epsilon)
    sigmas = torch.clamp(torch.exp(lmb), min=sigma_min, max=sigma_max)
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def karras_dynamic_scheduler(n, sigma_min, sigma_max, device="cpu"):
    """Karras et al. (2022), with rho oscillating along the schedule."""
    rho = 7.0
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = torch.zeros_like(ramp)
    for i in range(n):
        local_rho = math.cos(i * math.tau / n) * 2 + rho
        hi, lo = sigma_max ** (1 / local_rho), sigma_min ** (1 / local_rho)
        sigmas[i] = (hi + ramp[i] * (lo - hi)) ** local_rho
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def _custom_scheduler(n, sigma_min, sigma_max, device):
    """User-defined sigma curve, read from Settings (no generation-page UI).

    Two accepted forms:

    * a literal list ``[1.0, 0.85, ..., 0.0]`` - log-interpolated to ``n``;
    * an expression in ``x`` (0..1 progress), ``m`` (sigma_min), ``M``
      (sigma_max), plus ``phi`` and ``pi``.

    The expression is evaluated with no builtins and a whitelist of maths
    functions, so a malformed entry degrades to a linear ramp instead of
    executing arbitrary code.
    """
    expression = _custom_expression()

    if expression.startswith("[") and expression.endswith("]"):
        try:
            sigmas_list = [float(v) for v in expression.strip("[]").split(",")]
        except ValueError:
            return torch.cat([torch.linspace(sigma_max, sigma_min, n, device=device), torch.zeros(1, device=device)])
        if len(sigmas_list) < 2:
            return torch.cat([torch.linspace(sigma_max, sigma_min, n, device=device), torch.zeros(1, device=device)])
        if sigmas_list[0] == 1.0 and sigmas_list[-1] == 0.0:
            sigmas_list = [v * (sigma_max - sigma_min) + sigma_min for v in sigmas_list]
        xs = numpy.linspace(0, 1, len(sigmas_list))
        ys = numpy.log(numpy.clip(sigmas_list[::-1], 1e-8, None))
        new_ys = numpy.interp(numpy.linspace(0, 1, n), xs, ys)
        sigmas = torch.tensor(numpy.exp(new_ys)[::-1].copy(), device=device, dtype=torch.float32)
        return torch.cat([sigmas, sigmas.new_zeros([1])])

    sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)
    env = dict(_SAFE_EVAL_GLOBALS)
    env["phi"] = (1 + 5**0.5) / 2
    env["pi"] = math.pi
    env["m"] = float(sigma_min)
    env["M"] = float(sigma_max)
    try:
        import ast
        tree = ast.parse(expression, mode="eval")
        allowed = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp,
                   ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
                   ast.Mod, ast.USub, ast.UAdd, ast.Call)
        if len(expression) > 2000 or any(not isinstance(node, allowed) for node in ast.walk(tree)):
            raise ValueError("unsupported expression syntax")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_EVAL_GLOBALS):
                raise ValueError("unsupported function")
        code = compile(tree, "<pi-ss custom sigmas>", "eval")
        for s in range(n):
            env["x"] = (s / (n - 1)) if n > 1 else 0.0
            env["s"] = s
            env["n"] = n
            sigmas[s] = float(eval(code, env))  # noqa: S307 - whitelisted namespace
    except Exception as exc:
        from .. import warn

        warn(f"custom scheduler expression failed ({exc}); falling back to a linear ramp")
        sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def custom_scheduler(n, sigma_min, sigma_max, device):
    if n < 1:
        return torch.zeros(1, device=device)
    result = _custom_scheduler(n, sigma_min, sigma_max, device)
    if (torch.isfinite(result).all() and (result[:-1] > 0).all()
            and (result[:-1] >= result[1:]).all()):
        return result
    from .. import warn
    warn("custom scheduler must be finite, positive and descending; using a linear ramp")
    return torch.cat([torch.linspace(sigma_max, sigma_min, n, device=device), torch.zeros(1, device=device)])


# --------------------------------------------------------------------------
# RES4LYF family
# --------------------------------------------------------------------------
def tan_scheduler(n, sigma_min, sigma_max, device, *, pivot=0.6, slope=0.2):
    """RES4LYF single-stage tangent curve.

    Forge Neo already ships the two-stage ``bong_tangent``; this is the
    one-stage variant, which is a different curve rather than a duplicate.
    """
    steps = n + 2
    slope_eff = slope / (steps / 40.0)
    pivot_step = int(steps * pivot)

    smax = ((2 / math.pi) * math.atan(-slope_eff * (0 - pivot_step)) + 1) / 2
    smin = ((2 / math.pi) * math.atan(-slope_eff * ((steps - 1) - pivot_step)) + 1) / 2
    srange = smax - smin
    sscale = float(sigma_max) - float(sigma_min)

    sigmas = [
        ((((2 / math.pi) * math.atan(-slope_eff * (x - pivot_step)) + 1) / 2) - smin) * (1.0 / srange) * sscale
        + float(sigma_min)
        for x in range(steps)
    ]
    out = sigmas[:n]
    out.append(0.0)
    return torch.FloatTensor(out).to(device)


def beta57_scheduler(n, sigma_min, sigma_max, inner_model, device):
    """RES4LYF ``beta57``: Forge's Beta schedule with alpha/beta pinned to 0.5/0.7.

    Upstream is ``comfy.samplers.beta_scheduler(model_sampling, total_steps,
    alpha=0.5, beta=0.7)``.  Forge's stock ``beta`` reads alpha/beta from
    ``shared.opts``; this variant hard-pins the RES4LYF defaults so the curve
    does not drift when the user retunes the Beta sliders.
    """
    from scipy import stats

    alpha, beta = 0.5, 0.7
    total_timesteps = len(inner_model.sigmas) - 1
    ts = 1 - numpy.linspace(0, 1, n, endpoint=False)
    ts = numpy.rint(stats.beta.ppf(ts, alpha, beta) * total_timesteps)

    sigs: list[float] = []
    for t in ts:
        sigs.append(float(inner_model.sigmas[int(t)]))
    sigs.append(0.0)
    return torch.FloatTensor(sigs).to(device)


# --------------------------------------------------------------------------
# Registration table consumed by lib/register.py
#
# (name, label, function, need_inner_model, default_rho, aliases)
# --------------------------------------------------------------------------
SCHEDULER_TABLE = [
    ("cosine", "Cosine", cosine_scheduler, False, -1.0, None),
    ("cosexp", "CosineExponential blend", cosexpblend_scheduler, False, -1.0, None),
    ("phi", "Phi", phi_scheduler, False, -1.0, None),
    ("laplace", "Laplace", laplace_scheduler, False, -1.0, None),
    ("karras_dyn", "Karras Dynamic", karras_dynamic_scheduler, False, -1.0, None),
    ("custom", "Custom", custom_scheduler, False, -1.0, ["custom"]),
    ("tan", "Tan", tan_scheduler, False, -1.0, None),
    ("beta57", "Beta57", beta57_scheduler, True, -1.0, None),
]

#: Names deliberately NOT registered, with the reason. Surfaced in the log and
#: in ``inventory.json`` so the refusal is auditable rather than silent.
REFUSED = {}  # Linear Log is covered by Forge Exponential.
