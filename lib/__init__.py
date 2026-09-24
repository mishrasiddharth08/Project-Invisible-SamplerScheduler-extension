"""PROJECT INVISIBLE - SamplerScheduler: shared plumbing.

This package is deliberately loaded under a *unique* module name
(``pi_samplerscheduler_lib``) by ``scripts/engine.py`` instead of being put on
``sys.path`` as the generic name ``lib``.  Every WebUI extension has a
``scripts`` directory and many have a ``lib`` directory; whichever extension
loads first would otherwise win the name and the others would silently import
the wrong module.  Loading under a unique name is what lets this engine sit
next to every other extension without touching them.

Nothing in this package imports Forge at module scope.  Forge symbols are
resolved lazily and defensively so that a ``git pull`` on the ``neo`` branch
that moves or renames something degrades to "this feature is off" instead of a
traceback during startup.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

# --------------------------------------------------------------------------
# Identity / paths
# --------------------------------------------------------------------------
NAME = "PROJECT INVISIBLE - SamplerScheduler"
SHORT = "Invisible-SS"
VERSION = "1.1.0"

#: Extension root, i.e. ``extensions/project-invisible-samplerscheduler``.
EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_PATH = os.path.join(EXT_ROOT, "config.json")
MANIFEST_PATH = os.path.join(EXT_ROOT, "MANIFEST_OLD_FOLDERS.json")
INVENTORY_PATH = os.path.join(EXT_ROOT, "inventory.json")
PENDING_PATH = os.path.join(EXT_ROOT, "pending.json")
WRAPPERS_DIR = os.path.join(EXT_ROOT, "lib", "wrappers")


def forge_root() -> str:
    """Absolute path of the Forge Neo checkout that owns this extension."""
    return os.path.dirname(os.path.dirname(EXT_ROOT))


# --------------------------------------------------------------------------
# Logging - one prefix, matching the Ideogram-4 engine's house style.
# --------------------------------------------------------------------------
_QUIET = os.environ.get("PI_SS_QUIET", "").strip().lower() in {"1", "true", "yes"}


def log(msg: str) -> None:
    if not _QUIET:
        print(f"[{SHORT}] {msg}")


def warn(msg: str) -> None:
    print(f"[{SHORT}] WARNING: {msg}")


def error(msg: str) -> None:
    print(f"[{SHORT}] ERROR: {msg}")


def debug(msg: str) -> None:
    if os.environ.get("PI_SS_DEBUG", "").strip().lower() in {"1", "true", "yes"}:
        print(f"[{SHORT}] debug: {msg}")


# --------------------------------------------------------------------------
# JSON helpers - every read is total, every write is atomic.
# --------------------------------------------------------------------------
def read_json(path: str, default: Any = None) -> Any:
    """Read JSON, returning ``default`` on *any* problem.

    A corrupt or half-written config must never stop Forge from booting, so a
    parse failure is a warning and a fresh default, not an exception.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
    except Exception as exc:
        warn(f"could not read {os.path.basename(path)} ({exc}); using defaults")
        return default


def write_json(path: str, data: Any) -> bool:
    """Write JSON atomically (temp file + replace) so a crash can't truncate it.

    The temp name carries the process and thread id: the ComfyUI sync runs on a
    background thread while startup is still writing config, and on Windows a
    shared ``.tmp`` name makes ``os.replace`` fail with "file in use".
    """
    import threading

    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, sort_keys=False)
            fh.write("\n")
        os.replace(tmp, path)
        return True
    except Exception as exc:
        warn(f"could not write {os.path.basename(path)}: {exc}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DEFAULT_CONFIG: dict[str, Any] = {
    "_comment": (
        "PROJECT INVISIBLE - SamplerScheduler. Edit 'denylist' to hide names "
        "from the dropdowns without any UI. Delete this file to re-probe "
        "hardware and rebuild the inventory on next launch."
    ),
    "version": VERSION,
    "first_run": None,
    "last_run": None,
    "hardware": {},
    # Names this engine has ever seen registered (stock + ours). Used to spot
    # the case where Forge Neo later ships a name we also add, so stock wins.
    "seen_names": {"samplers": [], "schedulers": []},
    # Names this engine registered on the previous launch. Lets us unregister
    # exactly what we own and nothing else.
    "owned_names": {"samplers": [], "schedulers": []},
    "old_folders": [],
    "old_folders_state": "pending",  # pending | verified | disabled | deleted
    "denylist": {"samplers": [], "schedulers": []},
    "comfy_sync": {
        "enabled": True,
        "interval_hours": 168,
        "last_check": None,
        "last_comfy_sha": None,
        "auto_register_resolvable": False,
        "network_timeout_seconds": 10,
    },
    "custom_sigmas": "m + (M-m)*(1-x)**3",
}


def load_config() -> dict[str, Any]:
    """Load config.json, backfilling any key added by a newer version."""
    cfg = read_json(CONFIG_PATH, None)
    if not isinstance(cfg, dict):
        cfg = {}
        cfg.update(json.loads(json.dumps(DEFAULT_CONFIG)))
        cfg["first_run"] = now_iso()

    # Backfill missing keys so an old config survives an engine upgrade.
    for key, value in DEFAULT_CONFIG.items():
        if key not in cfg:
            cfg[key] = json.loads(json.dumps(value))  # deep copy
    for section in ("seen_names", "owned_names", "denylist"):
        if not isinstance(cfg.get(section), dict):
            cfg[section] = json.loads(json.dumps(DEFAULT_CONFIG[section]))
        for bucket in ("samplers", "schedulers"):
            if not isinstance(cfg[section].get(bucket), list):
                cfg[section][bucket] = []
    if not isinstance(cfg.get("comfy_sync"), dict):
        cfg["comfy_sync"] = json.loads(json.dumps(DEFAULT_CONFIG["comfy_sync"]))
    else:
        for key, value in DEFAULT_CONFIG["comfy_sync"].items():
            cfg["comfy_sync"].setdefault(key, value)

    cfg["version"] = VERSION
    return cfg


def save_config(cfg: dict[str, Any]) -> bool:
    cfg["last_run"] = now_iso()
    return write_json(CONFIG_PATH, cfg)


# --------------------------------------------------------------------------
# Package identity
# --------------------------------------------------------------------------
#: Unique module name this package is loaded under by ``scripts/engine.py``.
#: Never the bare name ``lib`` - see the module docstring for why.
PACKAGE_ALIAS = "pi_samplerscheduler_lib"

# ``sys`` is imported for submodules that re-export it via ``from . import sys``
# style access during teardown; keep the reference alive explicitly.
_sys = sys
