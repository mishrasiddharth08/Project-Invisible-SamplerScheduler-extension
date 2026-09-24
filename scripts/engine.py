"""PROJECT INVISIBLE: one startup entry point; no generation-page controls."""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
import traceback

_EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ALIAS = "pi_samplerscheduler_lib"

#: Re-entry guard. Forge re-imports ``scripts/*.py`` on a UI reload; the flag
#: lives on the (cached) lib module so it survives that re-import. Registration
#: is idempotent regardless, so this is purely to keep reloads instant.
_GUARD = "_pi_ss_startup_done"


def _bootstrap_lib():
    """Import ``lib/`` under a unique name, without touching ``sys.path``.

    Putting the extension root on ``sys.path`` would publish the generic names
    ``lib`` and ``scripts`` process-wide, where they would collide with every
    other extension that has those directories.  Loading by file location under
    a private alias avoids that entirely.
    """
    if _ALIAS in sys.modules:
        return sys.modules[_ALIAS]

    init_py = os.path.join(_EXT_ROOT, "lib", "__init__.py")
    spec = importlib.util.spec_from_file_location(
        _ALIAS,
        init_py,
        submodule_search_locations=[os.path.join(_EXT_ROOT, "lib")],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {init_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_ALIAS] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_ALIAS, None)
        raise
    return module


def _main():
    pi = _bootstrap_lib()
    if getattr(pi, _GUARD, False):
        return
    from pi_samplerscheduler_lib import comfy_sync, detect, register
    cfg = pi.load_config()
    # Only live object identities are owned. Saved names may now belong to Forge.
    register.unregister_all()
    denylist = cfg.get("denylist") or {}
    register.register_samplers(denylist.get("samplers"))
    register.register_schedulers(denylist.get("schedulers"))
    register.invalidate_caches()
    cfg["owned_names"] = register.owned()
    pi.log(f"v{pi.VERSION}: {register.summarise('samplers')} samplers; {register.summarise('schedulers')} schedulers")
    detect.write_inventory(register.owned())
    _install_callbacks(pi, cfg, register)
    pi.save_config(cfg)
    setattr(pi, _GUARD, True)
    if comfy_sync.should_check(cfg):
        def sync():
            try:
                comfy_sync.run(cfg)
                pi.save_config(cfg)
            except Exception as exc:
                pi.debug(f"ComfyUI check failed: {exc}")
        threading.Thread(target=sync, name="pi-ss-comfy-check", daemon=True).start()


def _install_callbacks(pi, cfg, register):
    from modules import script_callbacks
    def _on_ui_settings() -> None:
        try:
            import gradio as gr
            from modules import shared

            section = ("pi_samplerscheduler", "PROJECT INVISIBLE - SamplerScheduler")
            shared.opts.add_option(
                "pi_ss_custom_sigmas",
                shared.OptionInfo(
                    cfg.get("custom_sigmas", "m + (M-m)*(1-x)**3"),
                    "Custom scheduler: sigma expression or literal list",
                    gr.Textbox,
                    {"interactive": True},
                    section=section,
                ).info(
                    "Used only by the 'Custom' schedule type. Variables: x = 0..1 progress, "
                    "m = sigma_min, M = sigma_max, plus phi and pi. A literal list such as "
                    "[1.0, 0.7, 0.3, 0.0] is log-interpolated instead."
                ),
            )
        except Exception as exc:
            pi.debug(f"settings entry not added: {exc}")

    script_callbacks.on_ui_settings(_on_ui_settings)

    def unloaded():
        register.unregister_all()
        setattr(pi, _GUARD, False)
    script_callbacks.on_script_unloaded(unloaded)


try:
    _main()
except Exception as exc:
    print(f"[Invisible-SS] Could not complete registration: {exc}")
    traceback.print_exc()
