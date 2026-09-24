"""Tests that run without Forge, PyTorch, or a GPU.

Everything here exercises the parts of the engine that must work before any
host symbol is touched: the config loader, and the ComfyUI name-list parser.
Anything needing torch or a live Forge tree is out of scope for CI and is
covered by the manual acceptance run documented in the README.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALIAS = "pi_samplerscheduler_lib"


def load_lib():
    """Load lib/ the same way scripts/engine.py does: private alias, no sys.path."""
    if ALIAS in sys.modules:
        return sys.modules[ALIAS]
    spec = importlib.util.spec_from_file_location(
        ALIAS,
        os.path.join(REPO, "lib", "__init__.py"),
        submodule_search_locations=[os.path.join(REPO, "lib")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[ALIAS] = mod
    spec.loader.exec_module(mod)
    return mod


FIXTURE = '''
KSAMPLER_NAMES = ["euler", "euler_cfg_pp", "euler_ancestral", "heun", "heunpp2",
                  "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "ipndm",
                  "ipndm_v", "deis", "ddpm", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde"]

SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]

SCHEDULER_NAMES = ["simple", "sgm_uniform", "karras", "exponential",
                   "ddim_uniform", "beta", "normal", "linear_quadratic",
                   "kl_optimal"]
'''


def test_parse_name_lists():
    lib = load_lib()
    from pi_samplerscheduler_lib import comfy_sync

    got = comfy_sync.parse_name_lists(FIXTURE)
    assert "ipndm" in got["KSAMPLER_NAMES"], got["KSAMPLER_NAMES"]
    assert len(got["SCHEDULER_NAMES"]) == 9, got["SCHEDULER_NAMES"]
    # SAMPLER_NAMES is defined as KSAMPLER_NAMES + [...], which has no literal
    # list of its own to parse; the parser must fall back to KSAMPLER_NAMES
    # rather than returning nothing.
    assert got["SAMPLER_NAMES"], "fell through to an empty sampler list"
    assert lib.VERSION


def test_parse_survives_garbage():
    from pi_samplerscheduler_lib import comfy_sync

    for junk in ("", "nothing here", "SAMPLER_NAMES = some_call()"):
        got = comfy_sync.parse_name_lists(junk)
        assert isinstance(got, dict)
        assert all(isinstance(v, list) for v in got.values())


def test_prettify_matches_forge_style():
    from pi_samplerscheduler_lib import comfy_sync

    cases = {
        "dpmpp_2m_sde": "DPM++ 2M SDE",
        "euler_ancestral": "Euler a",
        "ipndm": "iPNDM",
        "dpmpp_2s_ancestral": "DPM++ 2S a",
        "euler_cfg_pp": "Euler CFG++",
    }
    for name, expected in cases.items():
        assert comfy_sync.prettify(name) == expected, (name, comfy_sync.prettify(name))


def test_config_roundtrip_and_backfill():
    lib = load_lib()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.json")
        original = lib.CONFIG_PATH
        try:
            lib.CONFIG_PATH = path
            # A config from an older version, missing most keys.
            lib.write_json(path, {"denylist": {"samplers": ["Euler A2"]}})
            cfg = lib.load_config()
            assert cfg["denylist"]["samplers"] == ["Euler A2"], "user setting lost"
            for key in ("seen_names", "owned_names", "comfy_sync", "custom_sigmas"):
                assert key in cfg, f"{key} not backfilled"
            assert lib.save_config(cfg)
            assert json.load(open(path, encoding="utf-8"))["version"] == lib.VERSION
        finally:
            lib.CONFIG_PATH = original


def test_corrupt_config_does_not_raise():
    lib = load_lib()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{ this is not json")
        assert lib.read_json(path, {"fallback": True}) == {"fallback": True}


def test_current_upstream_layout():
    load_lib()
    from pi_samplerscheduler_lib import comfy_sync
    got = comfy_sync.parse_name_lists('KSAMPLER_NAMES=["euler"]\nSAMPLER_NAMES=KSAMPLER_NAMES+["uni_pc_bh2"]\nSCHEDULER_HANDLERS={"beta": arbitrary_call()}\nSCHEDULER_NAMES=list(SCHEDULER_HANDLERS)')
    assert got["SAMPLER_NAMES"] == ["euler", "uni_pc_bh2"]
    assert got["SCHEDULER_NAMES"] == ["beta"]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'FAILED' if failures else 'OK'} - {failures} failure(s)")
    sys.exit(1 if failures else 0)
