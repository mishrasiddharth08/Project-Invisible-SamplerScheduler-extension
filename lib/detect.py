"""Detection layer: what does THIS Forge Neo tree already have?

Everything here is feature-detected.  We never assume a symbol exists; we look
for it, and if the ``neo`` branch has moved or renamed it we record that fact
in the inventory and carry on with a reduced feature set.  That is what makes a
``git pull`` on the host a non-event for this engine.

Scan order is fixed by spec:

1. ``modules/sd_samplers.py``            -> all_samplers / all_samplers_map
2. ``modules/sd_schedulers.py``          -> schedulers / schedulers_map
3. ``modules_forge/forge_alter_samplers.py`` -> samplers_data_alter
4. ``modules_forge/presets.py``          -> per-arch SAMPLERS / SCHEDULERS
5. ``extensions/*``                      -> leftover extra-sampler folders
"""

from __future__ import annotations

import os
import re
from typing import Any

from . import EXT_ROOT, INVENTORY_PATH, debug, forge_root, now_iso, warn, write_json

# --------------------------------------------------------------------------
# Host module handles. Resolved once, lazily, and never assumed present.
# --------------------------------------------------------------------------
_HOST: dict[str, Any] = {}


def host() -> dict[str, Any]:
    """Return a dict of the Forge symbols we could actually find.

    Keys that are missing simply do not appear.  Callers must use ``.get()``.
    """
    if _HOST:
        return _HOST

    caps: list[str] = []
    missing: list[str] = []

    # 1. modules/sd_samplers.py
    try:
        from modules import sd_samplers

        _HOST["sd_samplers"] = sd_samplers
        for attr in ("all_samplers", "all_samplers_map", "add_sampler", "set_samplers", "samplers", "samplers_for_img2img"):
            if hasattr(sd_samplers, attr):
                caps.append(f"sd_samplers.{attr}")
            else:
                missing.append(f"sd_samplers.{attr}")
    except Exception as exc:
        missing.append(f"modules.sd_samplers ({exc})")

    # 2. modules/sd_samplers_common.py -> SamplerData constructor
    try:
        from modules import sd_samplers_common

        _HOST["sd_samplers_common"] = sd_samplers_common
        if hasattr(sd_samplers_common, "SamplerData"):
            _HOST["SamplerData"] = sd_samplers_common.SamplerData
            caps.append("sd_samplers_common.SamplerData")
        else:
            missing.append("sd_samplers_common.SamplerData")
    except Exception as exc:
        missing.append(f"modules.sd_samplers_common ({exc})")

    # 3. modules/sd_samplers_kdiffusion.py -> KDiffusionSampler + extra params
    try:
        from modules import sd_samplers_kdiffusion

        _HOST["sd_samplers_kdiffusion"] = sd_samplers_kdiffusion
        if hasattr(sd_samplers_kdiffusion, "KDiffusionSampler"):
            _HOST["KDiffusionSampler"] = sd_samplers_kdiffusion.KDiffusionSampler
            caps.append("sd_samplers_kdiffusion.KDiffusionSampler")
        else:
            missing.append("sd_samplers_kdiffusion.KDiffusionSampler")
        if hasattr(sd_samplers_kdiffusion, "sampler_extra_params"):
            _HOST["sampler_extra_params"] = sd_samplers_kdiffusion.sampler_extra_params
            caps.append("sd_samplers_kdiffusion.sampler_extra_params")
        if hasattr(sd_samplers_kdiffusion, "samplers_data_k_diffusion"):
            caps.append("sd_samplers_kdiffusion.samplers_data_k_diffusion")
    except Exception as exc:
        missing.append(f"modules.sd_samplers_kdiffusion ({exc})")

    # 4. modules/sd_schedulers.py
    try:
        from modules import sd_schedulers

        _HOST["sd_schedulers"] = sd_schedulers
        if hasattr(sd_schedulers, "Scheduler"):
            _HOST["Scheduler"] = sd_schedulers.Scheduler
            caps.append("sd_schedulers.Scheduler")
        else:
            missing.append("sd_schedulers.Scheduler")
        for attr in ("schedulers", "schedulers_map", "all_schedulers"):
            if hasattr(sd_schedulers, attr):
                caps.append(f"sd_schedulers.{attr}")
            else:
                missing.append(f"sd_schedulers.{attr}")
    except Exception as exc:
        missing.append(f"modules.sd_schedulers ({exc})")

    # 5. Forge's own k_diffusion package (never vendor a second copy)
    try:
        import k_diffusion  # resolved via modules_forge/packages on sys.path

        _HOST["k_diffusion"] = k_diffusion
        caps.append("k_diffusion.sampling")
    except Exception as exc:
        missing.append(f"k_diffusion ({exc})")

    # 6. modules_forge/forge_alter_samplers.py (informational only)
    try:
        from modules_forge import forge_alter_samplers

        _HOST["forge_alter_samplers"] = forge_alter_samplers
        if hasattr(forge_alter_samplers, "samplers_data_alter"):
            caps.append("forge_alter_samplers.samplers_data_alter")
    except Exception:
        missing.append("modules_forge.forge_alter_samplers")

    # 7. modules_forge/presets.py (informational only - we never write to it)
    try:
        from modules_forge import presets

        _HOST["presets"] = presets
        caps.append("modules_forge.presets")
    except Exception:
        missing.append("modules_forge.presets")

    _HOST["_capabilities"] = caps
    _HOST["_missing"] = missing
    return _HOST


def can_register_samplers() -> bool:
    h = host()
    return all(k in h for k in ("sd_samplers", "SamplerData", "KDiffusionSampler"))


def can_register_schedulers() -> bool:
    h = host()
    return all(k in h for k in ("sd_schedulers", "Scheduler"))


# --------------------------------------------------------------------------
# Live registry snapshots
# --------------------------------------------------------------------------
def current_sampler_names() -> list[str]:
    """Every sampler label currently registered with Forge."""
    sd_samplers = host().get("sd_samplers")
    if sd_samplers is None:
        return []
    try:
        return [s.name for s in getattr(sd_samplers, "all_samplers", [])]
    except Exception:
        return []


def current_sampler_aliases() -> set[str]:
    """Every alias currently claimed by a registered sampler."""
    sd_samplers = host().get("sd_samplers")
    out: set[str] = set()
    if sd_samplers is None:
        return out
    try:
        for s in getattr(sd_samplers, "all_samplers", []):
            out.update(a for a in (s.aliases or []))
    except Exception:
        pass
    return out


def current_scheduler_names() -> list[str]:
    """Every scheduler ``name`` currently registered with Forge."""
    sd_schedulers = host().get("sd_schedulers")
    if sd_schedulers is None:
        return []
    try:
        pool = getattr(sd_schedulers, "all_schedulers", None) or getattr(sd_schedulers, "schedulers", [])
        return [s.name for s in pool]
    except Exception:
        return []


def current_scheduler_labels() -> list[str]:
    sd_schedulers = host().get("sd_schedulers")
    if sd_schedulers is None:
        return []
    try:
        pool = getattr(sd_schedulers, "all_schedulers", None) or getattr(sd_schedulers, "schedulers", [])
        return [s.label for s in pool]
    except Exception:
        return []


def k_diffusion_has(funcname: str) -> bool:
    """True when Forge's bundled k_diffusion already implements ``funcname``."""
    kd = host().get("k_diffusion")
    if kd is None:
        return False
    return hasattr(getattr(kd, "sampling", None), funcname)


# --------------------------------------------------------------------------
# Old-extension discovery
# --------------------------------------------------------------------------
#: Folders named by spec, plus anything else that registers samplers.
KNOWN_OLD_FOLDERS = (
    "sd_forge_neo_extra_samplers",
    "sd-forge-extra-samplers",
    "forge-beta57-scheduler",
    "webUI_ExtraSchedulers",
    "sd-forge-res4lyf",
)

#: Folders that must never be touched even if they match the heuristic.
NEVER_TOUCH = (
    "project-invisible-samplerscheduler",
    "project-invisible-ideogram-4",
    "Config-Presets",
)

_REGISTER_PATTERNS = (
    re.compile(r"\bSamplerData\s*\("),
    re.compile(r"\badd_sampler\s*\("),
    re.compile(r"\ball_samplers\s*\.\s*extend\b"),
    re.compile(r"\bschedulers\s*\.\s*append\s*\("),
    re.compile(r"\bsd_schedulers\.Scheduler\s*\("),
)


def _folder_registers_samplers(path: str) -> list[str]:
    """Return the evidence lines proving a folder registers samplers/schedulers."""
    evidence: list[str] = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git", "venv", "node_modules"}]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            fp = os.path.join(root, fn)
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
            except Exception:
                continue
            for pat in _REGISTER_PATTERNS:
                if pat.search(text):
                    rel = os.path.relpath(fp, path).replace("\\", "/")
                    evidence.append(f"{rel}: {pat.pattern}")
                    break
        if len(evidence) >= 8:
            break
    return evidence


def find_old_sampler_folders() -> list[dict[str, Any]]:
    """Locate leftover extra-sampler / extra-scheduler extension folders.

    Detection is by *behaviour* (does it call the registration API?), with the
    spec's named folders checked first.  Anything under :data:`NEVER_TOUCH` is
    skipped no matter what it contains.
    """
    ext_dir = os.path.join(forge_root(), "extensions")
    found: list[dict[str, Any]] = []
    if not os.path.isdir(ext_dir):
        return found

    try:
        entries = sorted(os.listdir(ext_dir))
    except Exception as exc:
        warn(f"cannot list extensions dir: {exc}")
        return found

    for entry in entries:
        path = os.path.join(ext_dir, entry)
        if not os.path.isdir(path) or entry in NEVER_TOUCH:
            continue
        known = entry in KNOWN_OLD_FOLDERS
        evidence = _folder_registers_samplers(path)
        if not (known and evidence) and not evidence:
            continue
        if not evidence:
            continue
        found.append(
            {
                "folder": entry,
                "path": path.replace("\\", "/"),
                "known": known,
                "evidence": evidence[:8],
                "disabled": os.path.exists(os.path.join(path, "disabled")),
            }
        )
    return found


# --------------------------------------------------------------------------
# Hardware probe (used only to decide whether GPU-only twins are worth adding)
# --------------------------------------------------------------------------
def probe_hardware() -> dict[str, Any]:
    """Best-effort hardware snapshot. Every field is optional."""
    info: dict[str, Any] = {"probed_at": now_iso()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info["gpu_name"] = props.name
            info["vram_total_mb"] = int(props.total_memory / (1024 * 1024))
            info["compute_capability"] = f"{props.major}.{props.minor}"
            try:
                free_b, _total_b = torch.cuda.mem_get_info(idx)
                info["vram_free_mb"] = int(free_b / (1024 * 1024))
            except Exception:
                pass
    except Exception as exc:
        debug(f"torch probe failed: {exc}")

    try:
        import psutil

        info["ram_total_mb"] = int(psutil.virtual_memory().total / (1024 * 1024))
    except Exception:
        pass

    # Which attention stack is already installed? We never install another one;
    # this is recorded purely so the README/inventory can show it.
    attention: list[str] = []
    for mod, label in (
        ("sageattention", "sage"),
        ("flash_attn", "flash"),
        ("xformers", "xformers"),
    ):
        try:
            __import__(mod)
            attention.append(label)
        except Exception:
            pass
    try:
        import torch

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            attention.append("sdpa")
    except Exception:
        pass
    info["attention_available"] = attention

    if info.get("ram_total_mb", 1 << 30) < 8192:
        info["ram_warning"] = "System RAM under 8 GB; sampling is unaffected but Forge may swap."
    return info


def hardware_changed(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """True when the GPU or driver changed enough to justify a re-probe."""
    keys = ("gpu_name", "vram_total_mb", "compute_capability", "torch")
    return any(old.get(k) != new.get(k) for k in keys)


# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------
def build_inventory(owned: dict[str, list[str]] | None = None) -> dict[str, Any]:
    h = host()
    owned = owned or {"samplers": [], "schedulers": []}
    # The classification report is the audit trail for every refusal, so it
    # belongs in the inventory rather than only in the startup log.
    try:
        from .register import report as _report

        classification = _report()
    except Exception:
        classification = {}
    sampler_names = current_sampler_names()
    scheduler_names = current_scheduler_names()
    return {
        "generated_at": now_iso(),
        "extension_root": EXT_ROOT.replace("\\", "/"),
        "forge_root": forge_root().replace("\\", "/"),
        "host_capabilities": h.get("_capabilities", []),
        "host_missing": h.get("_missing", []),
        "can_register_samplers": can_register_samplers(),
        "can_register_schedulers": can_register_schedulers(),
        "classification": classification,
        "samplers": {
            "total": len(sampler_names),
            "owned_by_engine": sorted(owned.get("samplers", [])),
            "stock_or_other": sorted(n for n in sampler_names if n not in set(owned.get("samplers", []))),
        },
        "schedulers": {
            "total": len(scheduler_names),
            "owned_by_engine": sorted(owned.get("schedulers", [])),
            "stock_or_other": sorted(n for n in scheduler_names if n not in set(owned.get("schedulers", []))),
        },
    }


def write_inventory(owned: dict[str, list[str]] | None = None) -> dict[str, Any]:
    inv = build_inventory(owned)
    write_json(INVENTORY_PATH, inv)
    return inv
