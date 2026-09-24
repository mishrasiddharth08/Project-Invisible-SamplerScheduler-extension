"""Bridge between Forge Neo's k-diffusion sampling loop and the RES4LYF trunks.

The vendored RES4LYF code under ``res4lyf/`` targets ComfyUI's API surface.
This module is the only place that knows how to hand it a Forge denoiser, and
it is imported lazily so that a failure here costs nothing at startup - the
RES/DEIS entries simply do not appear and every other sampler is unaffected.

The RES4LYF trunks were already ported and proven in the ``sd-forge-res4lyf``
extension; this is that port, moved in unchanged apart from the import paths.
"""

from __future__ import annotations

from typing import Callable

#: ``(label, rk_type, ode)`` - mirrors RES4LYF/beta/__init__.py.
RES_TRUNKS = [
    ("RES 2M", "res_2m", False),
    ("RES 3M", "res_3m", False),
    ("RES 2S", "res_2s", False),
    ("RES 3S", "res_3s", False),
    ("RES 5S", "res_5s", False),
    ("RES 6S", "res_6s", False),
    ("RES 2M ODE", "res_2m", True),
    ("RES 3M ODE", "res_3m", True),
    ("RES 2S ODE", "res_2s", True),
    ("RES 3S ODE", "res_3s", True),
    ("RES 5S ODE", "res_5s", True),
    ("RES 6S ODE", "res_6s", True),
    ("DEIS 2M", "deis_2m", False),
    ("DEIS 3M", "deis_3m", False),
    ("DEIS 2M ODE", "deis_2m", True),
    ("DEIS 3M ODE", "deis_3m", True),
]


def _load():
    """Import the vendored RES4LYF code, installing the comfy shim first."""
    from . import res4lyf as _pkg  # noqa: F401  (installs the additive comfy shim)
    from .res4lyf import _compat
    from .res4lyf.beta import rk_sampler_beta

    return _compat, rk_sampler_beta


def build_wrapper(rk_type: str, ode: bool = False) -> Callable:
    """Return a k-diffusion-shaped ``sample_*`` function for ``rk_type``.

    Raises on failure so the caller can log and skip this one trunk without
    losing the rest.
    """
    _compat, rk_beta = _load()
    sample_rk_beta = rk_beta.sample_rk_beta

    kwargs = dict(rk_type=rk_type, eta=0.0, eta_substep=0.0) if ode else dict(rk_type=rk_type)

    def _sample(model, x, sigmas, extra_args=None, callback=None, disable=None, **_ignored):
        # Give the vendored code the ComfyUI attribute layout it expects
        # (``model.inner_model.inner_model.model_sampling`` and friends).
        adapter = _compat.adapt_denoiser(model)

        # The vendored sampler expects ``extra_args`` to already carry
        # ``model_options['transformer_options']`` for region/mask plumbing.
        extra_args = {} if extra_args is None else dict(extra_args)
        model_options = dict(extra_args.get("model_options") or {})
        transformer_options = dict(model_options.get("transformer_options") or {})
        model_options["transformer_options"] = transformer_options
        extra_args["model_options"] = model_options

        # RES4LYF promotes its internal maths to fp64. Cast the result back to
        # the dtype/device Forge handed us so VAE decode and hires fix never
        # see an unexpected fp64 tensor (this is also the memory-safe choice:
        # the fp64 buffer dies with this frame instead of travelling onward).
        in_dtype, in_device = x.dtype, x.device
        out = sample_rk_beta(
            adapter,
            x,
            sigmas,
            None,  # sampler arg (unused by the beta trunks)
            extra_args,
            callback,
            disable,
            **kwargs,
        )
        if out.dtype != in_dtype or out.device != in_device:
            out = out.to(dtype=in_dtype, device=in_device)
        return out

    _sample.__name__ = f"sample_{rk_type}{'_ode' if ode else ''}"
    _sample.__qualname__ = _sample.__name__
    return _sample
