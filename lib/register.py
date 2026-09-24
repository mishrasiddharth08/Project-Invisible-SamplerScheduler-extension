"""Registration: the only module that writes to Forge's sampler/scheduler lists.

Design rules that make this safe to run next to a moving host:

* **Public API first.** Samplers go in through ``sd_samplers.add_sampler``,
  which is Forge's own entry point and already refuses duplicates.
* **Callables, not monkey-patching.** Forge's ``KDiffusionSampler`` accepts a
  callable in place of a ``sample_*`` attribute name (its own ``Restart`` and
  ``UniPC`` entries do exactly that), so this engine never assigns anything
  onto ``k_diffusion.sampling``.  Nothing global is mutated.
* **Stock always wins.** A candidate whose label *or* any alias is already
  claimed is dropped, logged, and never re-tried.  That is also the mechanism
  by which a future Forge Neo release that ships one of these names silently
  takes over from the extension copy.
* **Every registration is recorded.** ``unregister_all`` removes exactly the
  names this engine added and nothing else.
* **One failure costs one sampler.** Each candidate is resolved inside its own
  try/except, so a broken wrap is refused and logged rather than taking down
  the rest of the engine.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable, NamedTuple

from . import debug, warn
from .detect import (
    can_register_samplers,
    can_register_schedulers,
    current_sampler_aliases,
    current_sampler_names,
    current_scheduler_labels,
    current_scheduler_names,
    host,
)

# Names this engine added during THIS process, so teardown is exact.
_OWNED: dict[str, list[str]] = {"samplers": [], "schedulers": []}
#: Classification report for inventory.json / the console summary.
_OBJECTS = {"samplers": [], "schedulers": []}
_EXTRA_PARAMS = {}

_REPORT: dict[str, list[str]] = {
    "registered": [],
    "already_in_neo": [],
    "denied_by_config": [],
    "refused": [],
    "trunk_locked": [],
}


class Candidate(NamedTuple):
    """One sampler waiting to be registered."""

    label: str
    func_name: str
    loader: Callable[[], Callable]
    aliases: list[str]
    options: dict[str, Any]
    extra_params: list[str] | None
    source: str
    #: Label of the base sampler a CFG++ variant wraps. When set, the variant
    #: is only registered if that base is present (spec: trunk lock).
    requires_base: str | None = None


# --------------------------------------------------------------------------
# Candidate tables
# --------------------------------------------------------------------------
def _lazy(module_attr: str) -> Callable[[], Callable]:
    """Return a loader that imports ``lib.wrappers.<module>`` on first use.

    Deferring the import keeps startup cheap and confines an ImportError to
    the single sampler that needs it.
    """
    mod_name, attr = module_attr.rsplit(".", 1)

    def _load() -> Callable:
        import importlib

        mod = importlib.import_module(f"{__package__}.wrappers.{mod_name}")
        return getattr(mod, attr)

    return _load


def _lazy_needs(module_attr: str, *required: str) -> Callable[[], Callable]:
    """Like :func:`_lazy`, but proves ``required`` modules import first.

    Used where a sampler has a data dependency its own module loads lazily -
    ``DEIS`` needs the coefficient tables that live in the RES4LYF tree. Without
    this probe a build with that tree deleted would register DEIS happily and
    only fail once the user pressed Generate. Failing at registration turns a
    mid-generation crash into a logged refusal.
    """
    inner = _lazy(module_attr)

    def _load() -> Callable:
        import importlib

        for mod in required:
            importlib.import_module(f"{__package__}.wrappers.{mod}")
        return inner()

    return _load


def _res4lyf_loader(rk_type: str, ode: bool) -> Callable[[], Callable]:
    def _load() -> Callable:
        import importlib

        bridge = importlib.import_module(f"{__package__}.wrappers.res4lyf_bridge")
        return bridge.build_wrapper(rk_type, ode)

    return _load


_CHURN = ["s_churn", "s_tmin", "s_tmax", "s_noise"]


#: Upstream names deliberately NOT ported, with the reason. Recorded in the
#: startup log and in inventory.json so every refusal is auditable.
REFUSED_SAMPLERS = {}


def sampler_candidates() -> list[Candidate]:
    """Every sampler this engine can contribute, in registration order."""
    out: list[Candidate] = []

    # --- merged from sd_forge_neo_extra_samplers (Panchovix) --------------
    # The second alias, where it differs, is the *upstream ComfyUI* spelling.
    # Carrying it does two jobs: infotext written by ComfyUI resolves here, and
    # the ComfyUI sync recognises the name as already covered instead of
    # queueing a port for something this engine already provides.
    src = "sd_forge_neo_extra_samplers"
    for label, fn, aliases, options in (
        ("Gradient Estimation", "extra_samplers.sample_gradient_estimation", ["gradient_estimation"], {}),
        ("Gradient Estimation CFG++", "extra_samplers.sample_gradient_estimation_cfg_pp", ["gradient_estimation_cfg_pp"], {}),
        ("SEEDS 2", "extra_samplers.sample_seeds_2", ["seeds_2"], {}),
        ("SEEDS 3", "extra_samplers.sample_seeds_3", ["seeds_3"], {}),
        ("SA Solver", "extra_samplers.sample_sa_solver", ["sa_solver"], {}),
        ("SA Solver PECE", "extra_samplers.sample_sa_solver_pece", ["sa_solver_pece"], {}),
        ("DPM++ SDE CFG++", "extra_samplers.sample_dpmpp_sde_cfg_pp", ["dpmpp_sde_cfg_pp"], {}),
        ("EXP Heun 2 x0", "extra_samplers.sample_exp_heun_2_x0", ["exp_heun_2_x0"], {}),
        ("EXP Heun 2 x0 SDE", "extra_samplers.sample_exp_heun_2_x0_sde", ["exp_heun_2_x0_sde"], {}),
        ("Res Multistep CFG++", "extra_samplers.sample_res_multistep_cfg_pp", ["res_multistep_cfg_pp"], {}),
        ("Res Multistep A", "extra_samplers.sample_res_multistep_ancestral", ["res_multistep_aa", "res_multistep_ancestral"], {}),
        ("Res Multistep A CFG++", "extra_samplers.sample_res_multistep_ancestral_cfg_pp", ["res_multistep_a_cfg_pp", "res_multistep_ancestral_cfg_pp"], {}),
        ("Euler A2", "extra_samplers.sample_euler_a2", ["euler_a2"], {}),
    ):
        func_name = fn.split(".", 1)[1]
        base = None
        if label.endswith("CFG++"):
            base = {
                "Gradient Estimation CFG++": "Gradient Estimation",
                "DPM++ SDE CFG++": "DPM++ SDE",
                "Res Multistep CFG++": "Res Multistep",
                "Res Multistep A CFG++": "Res Multistep A",
            }.get(label)
        out.append(Candidate(label, func_name, _lazy(fn), list(aliases), options, None, src, base))

    # --- merged from webUI_ExtraSchedulers ------------------------------
    src = "webUI_ExtraSchedulers"
    out.append(
        Candidate(
            "Refined Exponential Solver",
            "sample_res_solver",
            _lazy("res_solver.sample_res_solver"),
            ["k_res"],
            {},
            None,
            src,
        )
    )
    out.append(
        Candidate(
            "DPM++ 4M SDE",
            "sample_clyb_4m_sde_momentumized",
            _lazy("clyb_4m_sde.sample_clyb_4m_sde_momentumized"),
            ["k_dpmpp_4m_sde"],
            {},
            None,
            src,
        )
    )
    for label, fn, alias, extra, base in (
        ("Euler a CFG++", "samplers_cfgpp.sample_euler_ancestral_cfgpp", "k_euler_a_cfgpp", None, "Euler a"),
        ("Euler CFG++", "samplers_cfgpp.sample_euler_cfgpp", "k_euler_cfgpp", _CHURN, "Euler"),
        ("Euler Dy CFG++", "samplers_cfgpp.sample_euler_dy_cfgpp", "k_euler_dy_cfgpp", _CHURN, "Euler"),
        ("Euler SMEA Dy CFG++", "samplers_cfgpp.sample_euler_smea_dy_cfgpp", "k_euler_smea_dy_cfgpp", _CHURN, "Euler"),
    ):
        options = {"uses_ensd": True} if label == "Euler a CFG++" else {}
        out.append(Candidate(label, fn.split(".", 1)[1], _lazy(fn), [alias], options, extra, src, base))

    # --- ported from ComfyUI master -------------------------------------
    # Forge Neo bundles a trimmed k-diffusion (19 sample_* functions against
    # ComfyUI's 46). These are the missing ones whose maths is self-contained
    # enough to port verbatim; see lib/wrappers/comfy_ported.py for the four
    # host-API adaptations and the upstream licences.
    src = "ComfyUI master (ported)"
    _ETA_NOISE = ["eta", "s_noise"]
    for label, fn, aliases, options, extra in (
        ("iPNDM", "comfy_ported.sample_ipndm", ["ipndm"], {}, None),
        ("iPNDM V", "comfy_ported.sample_ipndm_v", ["ipndm_v"], {}, None),
        # DEIS alone needs the RES4LYF coefficient tables; probe them so a
        # build without that tree refuses DEIS instead of crashing on Generate.
        ("DEIS", "comfy_ported.sample_deis", ["deis"], {}, None),
        ("DDPM", "comfy_ported.sample_ddpm", ["ddpm"], {}, None),
        ("Heun++2", "comfy_ported.sample_heunpp2", ["heunpp2", "k_heunpp2"], {"second_order": True}, _CHURN),
        (
            "DPM2 a",
            "comfy_ported.sample_dpm_2_ancestral",
            ["dpm_2_ancestral", "k_dpm_2_a", "k_dpm_2_ancestral"],
            # Matches Forge's own DPM2 entry: Karras by default, drops the
            # penultimate sigma, and counts as a second-order sampler so the
            # step estimate stays honest.
            {"scheduler": "karras", "discard_next_to_last_sigma": True, "second_order": True, "uses_ensd": True},
            _ETA_NOISE,
        ),
        ("DPM fast", "comfy_ported.sample_dpm_fast", ["dpm_fast", "k_dpm_fast"], {}, _ETA_NOISE),
        ("DPM adaptive", "comfy_ported.sample_dpm_adaptive", ["dpm_adaptive", "k_dpm_ad"], {}, _ETA_NOISE),
    ):
        loader = (
            _lazy_needs(fn, "res4lyf.beta.deis_coefficients")
            if label == "DEIS"
            else _lazy(fn)
        )
        out.append(Candidate(label, fn.split(".", 1)[1], loader, list(aliases), options, extra, src, None))

    # DPM++ 2M SDE Heun needs no new code at all: it is Forge's own
    # ``sample_dpmpp_2m_sde`` with solver_type="heun", which Forge already
    # threads through from SamplerData.options. Passing the function *name*
    # also means it inherits Forge's registered eta/s_noise extra params.
    out.append(
        Candidate(
            "DPM++ 2M SDE Heun",
            "sample_dpmpp_2m_sde",
            lambda: "sample_dpmpp_2m_sde",
            ["dpmpp_2m_sde_heun", "k_dpmpp_2m_sde_heun"],
            {"scheduler": "exponential", "brownian_noise": True, "solver_type": "heun"},
            None,
            "Forge k_diffusion (options-only variant)",
            None,
        )
    )

    # Reviewed ComfyUI additions are always available, including offline.
    for label, fn, aliases, options, extra in (
        ("DPM++ 2S a", "sample_dpmpp_2s_ancestral", ["dpmpp_2s_ancestral", "k_dpmpp_2s_a"], {"second_order": True, "uses_ensd": True}, ["eta", "s_noise"]),
        ("DPM++ 2S a CFG++", "comfy_latest.sample_dpmpp_2s_ancestral_cfg_pp", ["dpmpp_2s_ancestral_cfg_pp"], {"second_order": True, "uses_ensd": True}, ["eta", "s_noise"]),
        ("CFG++ UD10 AB", "comfy_latest.sample_cfgpp_ud10_ab", ["cfgpp_ud10_ab"], {}, None),
        ("UniPC BH2", "comfy_latest.sample_unipc_bh2", ["uni_pc_bh2"], {"discard_next_to_last_sigma": True}, None),
    ):
        loader = _lazy(fn) if "." in fn else lambda fn=fn: fn
        out.append(Candidate(label, fn.rsplit(".", 1)[-1], loader, aliases, options, extra, "ComfyUI reviewed 2026-09-11"))

    # --- merged from sd-forge-res4lyf (ClownsharkBatwing port) -----------
    src = "sd-forge-res4lyf"
    try:
        import importlib

        bridge = importlib.import_module(f"{__package__}.wrappers.res4lyf_bridge")
        trunks = bridge.RES_TRUNKS
    except Exception as exc:
        debug(f"RES4LYF trunk table unavailable: {exc}")
        trunks = []
    for label, rk_type, ode in trunks:
        func_name = f"sample_{rk_type}{'_ode' if ode else ''}"
        out.append(
            Candidate(
                f"{label} (RES4LYF)",
                func_name,
                _res4lyf_loader(rk_type, ode),
                [func_name, f"k_{func_name}"],
                # The RES trunks manage their own noise sampler internally, so
                # brownian_noise is deliberately NOT set - Forge would pass a
                # noise_sampler kwarg the wrapper has no use for.
                {"scheduler": "beta"},
                None,
                src,
            )
        )

    return out


# --------------------------------------------------------------------------
# Sampler registration
# --------------------------------------------------------------------------
def register_samplers(denylist: list[str] | None = None) -> list[str]:
    """Register every eligible sampler. Returns the labels actually added."""
    denylist = set(denylist or [])
    if not can_register_samplers():
        warn("Forge's sampler registry was not found; no samplers registered (host may have moved symbols)")
        return []

    h = host()
    sd_samplers = h["sd_samplers"]
    SamplerData = h["SamplerData"]
    KDiffusionSampler = h["KDiffusionSampler"]
    extra_params_map = h.get("sampler_extra_params")

    for name, reason in REFUSED_SAMPLERS.items():
        entry = f"{name}: {reason}"
        if entry not in _REPORT["refused"]:
            _REPORT["refused"].append(entry)

    existing_labels = set(current_sampler_names())
    existing_aliases = current_sampler_aliases()
    added: list[str] = []

    for cand in sampler_candidates():
        if {cand.label, cand.func_name, *cand.aliases} & denylist:
            _REPORT["denied_by_config"].append(cand.label)
            continue

        # Stock (or another extension) already owns this name -> stand down.
        if cand.label in existing_labels:
            _REPORT["already_in_neo"].append(cand.label)
            continue
        clash = [a for a in cand.aliases if a in existing_aliases]
        if clash:
            warn(f"name collision: '{cand.label}' alias {clash} already registered; keeping the existing one")
            _REPORT["already_in_neo"].append(cand.label)
            continue

        # Trunk lock: a CFG++ variant only exists if its base sampler does.
        if cand.requires_base and cand.requires_base not in existing_labels:
            _REPORT["trunk_locked"].append(f"{cand.label} (base '{cand.requires_base}' absent)")
            continue

        # Resolve the implementation. A broken wrap is refused here.
        try:
            func = cand.loader()
            if isinstance(func, str):
                # A plain name means "reuse Forge's own k_diffusion function".
                # Verify it really exists before claiming a dropdown slot.
                from .detect import k_diffusion_has

                if not k_diffusion_has(func):
                    raise AttributeError(f"Forge's k_diffusion has no '{func}'")
            elif not callable(func):
                raise TypeError(f"{cand.func_name} resolved to {type(func).__name__}, not a callable")
        except Exception as exc:
            _REPORT["refused"].append(f"{cand.label}: {exc}")
            warn(f"refused '{cand.label}' from {cand.source}: {exc}")
            debug(traceback.format_exc())
            continue

        # Forge accepts a callable in place of a k_diffusion attribute name, so
        # nothing is written into k_diffusion.sampling.
        if cand.extra_params and isinstance(extra_params_map, dict):
            _EXTRA_PARAMS.setdefault(func, (func in extra_params_map, extra_params_map.get(func)))
            extra_params_map[func] = list(cand.extra_params)

        data = SamplerData(
            cand.label,
            lambda model, _f=func: KDiffusionSampler(_f, model),
            list(cand.aliases),
            dict(cand.options),
        )
        try:
            sd_samplers.add_sampler(data)
        except Exception as exc:
            _REPORT["refused"].append(f"{cand.label}: add_sampler failed: {exc}")
            warn(f"add_sampler('{cand.label}') failed: {exc}")
            debug(traceback.format_exc())
            continue

        existing_labels.add(cand.label)
        existing_aliases.update(cand.aliases)
        added.append(cand.label)
        _OBJECTS["samplers"].append(data)
        _OWNED["samplers"].append(cand.label)
        _REPORT["registered"].append(cand.label)

    return added


# --------------------------------------------------------------------------
# Scheduler registration
# --------------------------------------------------------------------------
def register_schedulers(denylist: list[str] | None = None) -> list[str]:
    """Register every eligible scheduler. Returns the names actually added."""
    denylist = set(denylist or [])
    if not can_register_schedulers():
        warn("Forge's scheduler registry was not found; no schedulers registered")
        return []

    h = host()
    sd_schedulers = h["sd_schedulers"]
    Scheduler = h["Scheduler"]

    try:
        from .wrappers import schedulers as tbl
    except Exception as exc:
        warn(f"scheduler table unavailable: {exc}")
        return []

    for name, reason in getattr(tbl, "REFUSED", {}).items():
        _REPORT["refused"].append(f"scheduler '{name}': {reason}")

    full_pool = getattr(sd_schedulers, "all_schedulers", sd_schedulers.schedulers)
    existing_names = {s.name for s in full_pool}
    existing_labels = {s.label for s in full_pool}
    added: list[str] = []

    # ``hide_schedulers`` filters ``all_schedulers`` into ``schedulers`` at host
    # import time. Append to both so the user's hide list keeps working for our
    # entries exactly as it does for stock ones.
    hidden: set[str] = set()
    try:
        from modules import shared

        hidden = set(getattr(shared.opts, "hide_schedulers", []) or [])
    except Exception:
        pass

    for name, label, func, need_inner, default_rho, aliases in tbl.SCHEDULER_TABLE:
        if name in denylist or label in denylist:
            _REPORT["denied_by_config"].append(f"scheduler:{label}")
            continue
        if name in existing_names or label in existing_labels:
            _REPORT["already_in_neo"].append(f"scheduler:{label}")
            continue

        try:
            sch = Scheduler(name, label, func)
            # Optional dataclass fields - set defensively so a host that drops
            # or renames one of them does not break registration.
            if hasattr(sch, "need_inner_model"):
                sch.need_inner_model = need_inner
            if hasattr(sch, "default_rho"):
                sch.default_rho = default_rho
            if hasattr(sch, "aliases"):
                sch.aliases = list(aliases) if aliases else None
        except Exception as exc:
            _REPORT["refused"].append(f"scheduler '{label}': {exc}")
            warn(f"refused scheduler '{label}': {exc}")
            continue

        try:
            pool_all = getattr(sd_schedulers, "all_schedulers", None)
            if isinstance(pool_all, list) and pool_all is not sd_schedulers.schedulers:
                pool_all.append(sch)
            if label not in hidden:
                sd_schedulers.schedulers.append(sch)
            sd_schedulers.schedulers_map = {
                **{x.name: x for x in sd_schedulers.schedulers},
                **{x.label: x for x in sd_schedulers.schedulers},
            }
        except Exception as exc:
            _REPORT["refused"].append(f"scheduler '{label}': {exc}")
            warn(f"could not append scheduler '{label}': {exc}")
            continue

        existing_names.add(name)
        existing_labels.add(label)
        added.append(name)
        _OBJECTS["schedulers"].append(sch)
        _OWNED["schedulers"].append(name)
        _REPORT["registered"].append(f"scheduler:{label}")

    return added


# --------------------------------------------------------------------------
# Cache invalidation
# --------------------------------------------------------------------------
def invalidate_caches() -> None:
    """Clear Forge's memoised sampler/scheduler resolution.

    ``sd_samplers.get_sampler_and_scheduler`` is decorated with
    ``functools.cache``.  Anything resolved before we registered would keep its
    stale answer - including infotext parsing that must map our new names back
    onto the right sampler.
    """
    h = host()
    sd_samplers = h.get("sd_samplers")
    if sd_samplers is None:
        return
    fn = getattr(sd_samplers, "get_sampler_and_scheduler", None)
    if fn is not None and hasattr(fn, "cache_clear"):
        try:
            fn.cache_clear()
            debug("cleared get_sampler_and_scheduler cache")
        except Exception as exc:
            debug(f"cache_clear failed: {exc}")
    try:
        sd_samplers.set_samplers()
    except Exception as exc:
        debug(f"set_samplers() failed: {exc}")


# --------------------------------------------------------------------------
# Teardown
# --------------------------------------------------------------------------
def unregister_all(owned: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Remove exactly the names this engine added, restoring the stock lists.

    Used when the extension is disabled, and defensively at load time to clear
    a half-registered state left by an interrupted previous run.
    """
    owned = owned or _OWNED
    removed: dict[str, list[str]] = {"samplers": [], "schedulers": []}
    h = host()

    sd_samplers = h.get("sd_samplers")
    sampler_names = set(owned.get("samplers", []))
    if sd_samplers is not None and sampler_names:
        try:
            keep = [s for s in sd_samplers.all_samplers if not any(s is o for o in _OBJECTS["samplers"])]
            removed["samplers"] = [s.name for s in sd_samplers.all_samplers if any(s is o for o in _OBJECTS["samplers"])]
            sd_samplers.all_samplers[:] = keep
            sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
            sd_samplers.set_samplers()
        except Exception as exc:
            warn(f"could not fully unregister samplers: {exc}")

    sd_schedulers = h.get("sd_schedulers")
    scheduler_names = set(owned.get("schedulers", []))
    if sd_schedulers is not None and scheduler_names:
        try:
            for attr in ("all_schedulers", "schedulers"):
                pool = getattr(sd_schedulers, attr, None)
                if isinstance(pool, list):
                    gone = [s.name for s in pool if any(s is o for o in _OBJECTS["schedulers"])]
                    pool[:] = [s for s in pool if not any(s is o for o in _OBJECTS["schedulers"])]
                    if attr == "schedulers":
                        removed["schedulers"] = gone
            sd_schedulers.schedulers_map = {
                **{x.name: x for x in sd_schedulers.schedulers},
                **{x.label: x for x in sd_schedulers.schedulers},
            }
        except Exception as exc:
            warn(f"could not fully unregister schedulers: {exc}")

    _OWNED["samplers"] = [n for n in _OWNED["samplers"] if n not in sampler_names]
    _OWNED["schedulers"] = [n for n in _OWNED["schedulers"] if n not in scheduler_names]
    for key, (present, old) in _EXTRA_PARAMS.items():
        params = h.get("sampler_extra_params", {})
        if present:
            params[key] = old
        else:
            params.pop(key, None)
    _EXTRA_PARAMS.clear()
    for bucket in _OBJECTS:
        _OBJECTS[bucket].clear()
    for bucket in _REPORT:
        _REPORT[bucket].clear()
    invalidate_caches()
    return removed


def owned() -> dict[str, list[str]]:
    return {"samplers": list(_OWNED["samplers"]), "schedulers": list(_OWNED["schedulers"])}


def report() -> dict[str, list[str]]:
    return {k: list(v) for k, v in _REPORT.items()}


def summarise(kind: str = "samplers") -> str:
    """One-line tally for ``kind`` ("samplers" or "schedulers")."""
    want_sched = kind == "schedulers"

    def _count(bucket: str) -> int:
        return sum(1 for e in _REPORT[bucket] if e.startswith("scheduler") is want_sched)

    parts = [f"{_count('registered')} added"]
    for bucket, word in (
        ("already_in_neo", "already in Neo"),
        ("trunk_locked", "trunk-locked"),
        ("denied_by_config", "denied by config"),
        ("refused", "refused"),
    ):
        n = _count(bucket)
        if n:
            parts.append(f"{n} {word}")
    return ", ".join(parts)
