"""Source provenance and hash verification for the consolidated extension."""
from __future__ import annotations

import hashlib
import os
from typing import Any

from . import (
    MANIFEST_PATH,
    WRAPPERS_DIR,
    forge_root,
    log,
    now_iso,
    read_json,
    warn,
    write_json,
)
from .detect import NEVER_TOUCH, find_old_sampler_folders

#: Which merged file in ``lib/wrappers`` came from which old-folder file.
#: (old_folder, source_relative_path, our_path_relative_to_wrappers, modified)
PROVENANCE = [
    ("sd_forge_neo_extra_samplers", "scripts/extra_samplers.py", "extra_samplers.py", True),
    ("sd_forge_neo_extra_samplers", "scripts/sa_solver.py", "sa_solver.py", False),
    ("webUI_ExtraSchedulers", "scripts/res_solver.py", "res_solver.py", False),
    ("webUI_ExtraSchedulers", "scripts/samplers_cfgpp.py", "samplers_cfgpp.py", False),
    ("webUI_ExtraSchedulers", "scripts/clybius_dpmpp_4m_sde.py", "clyb_4m_sde.py", False),
    ("webUI_ExtraSchedulers", "scripts/extra_schedulers.py", "schedulers.py", True),
    ("sd-forge-res4lyf", "lib_res4lyf/beta/rk_sampler_beta.py", "res4lyf/beta/rk_sampler_beta.py", False),
    ("sd-forge-res4lyf", "lib_res4lyf/beta/rk_method_beta.py", "res4lyf/beta/rk_method_beta.py", False),
    ("sd-forge-res4lyf", "lib_res4lyf/beta/rk_coefficients_beta.py", "res4lyf/beta/rk_coefficients_beta.py", False),
    ("sd-forge-res4lyf", "lib_res4lyf/beta/rk_guide_func_beta.py", "res4lyf/beta/rk_guide_func_beta.py", False),
    ("sd-forge-res4lyf", "lib_res4lyf/beta/rk_noise_sampler_beta.py", "res4lyf/beta/rk_noise_sampler_beta.py", False),
    ("sd-forge-res4lyf", "lib_res4lyf/beta/noise_classes.py", "res4lyf/beta/noise_classes.py", False),
    ("sd-forge-res4lyf", "lib_res4lyf/beta/phi_functions.py", "res4lyf/beta/phi_functions.py", False),
    ("sd-forge-res4lyf", "lib_res4lyf/beta/deis_coefficients.py", "res4lyf/beta/deis_coefficients.py", False),
    ("sd-forge-res4lyf", "lib_res4lyf/beta/constants.py", "res4lyf/beta/constants.py", False),
    ("sd-forge-res4lyf", "lib_res4lyf/helper.py", "res4lyf/helper.py", False),
    ("sd-forge-res4lyf", "lib_res4lyf/latents.py", "res4lyf/latents.py", False),
    ("sd-forge-res4lyf", "lib_res4lyf/models.py", "res4lyf/models.py", False),
    ("sd-forge-res4lyf", "lib_res4lyf/sigmas.py", "res4lyf/sigmas.py", False),
    ("sd-forge-res4lyf", "lib_res4lyf/style_transfer.py", "res4lyf/style_transfer.py", False),
    ("sd-forge-res4lyf", "lib_res4lyf/res4lyf.py", "res4lyf/res4lyf.py", False),
    ("sd-forge-res4lyf", "lib_res4lyf/_compat.py", "res4lyf/_compat.py", True),
    ("sd-forge-res4lyf", "scripts/forge_res4lyf.py", "res4lyf_bridge.py", True),
]


def sha256(path: str) -> str | None:
    """SHA-256 of a file, or None if it cannot be read."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------
def build_manifest() -> dict[str, Any]:
    """Record every old folder found and the hash pairing for its merged files."""
    ext_dir = os.path.join(forge_root(), "extensions")
    detected = find_old_sampler_folders()
    detected_names = {d["folder"] for d in detected}

    folders: dict[str, Any] = {}
    for folder, src_rel, our_rel, modified in PROVENANCE:
        src_abs = os.path.join(ext_dir, folder, src_rel)
        our_abs = os.path.join(WRAPPERS_DIR, our_rel)
        entry = folders.setdefault(
            folder,
            {
                "path": os.path.join(ext_dir, folder).replace("\\", "/"),
                "present_on_disk": os.path.isdir(os.path.join(ext_dir, folder)),
                "detected_as_sampler_source": folder in detected_names,
                "files": [],
            },
        )
        entry["files"].append(
            {
                "source": src_rel,
                "merged_to": f"lib/wrappers/{our_rel}",
                "source_sha256": sha256(src_abs),
                "merged_sha256": sha256(our_abs),
                # A modified copy has intentional edits (import paths rewritten,
                # registration code removed) so the two hashes must differ.
                "modified_during_merge": modified,
                "source_exists": os.path.isfile(src_abs),
                "merged_exists": os.path.isfile(our_abs),
            }
        )

    # Any *other* folder the detector flagged: recorded, never auto-deleted,
    # because we hold no merged copy of its code.
    for d in detected:
        if d["folder"] in folders:
            folders[d["folder"]]["evidence"] = d["evidence"]
            continue
        folders[d["folder"]] = {
            "path": d["path"],
            "present_on_disk": True,
            "detected_as_sampler_source": True,
            "evidence": d["evidence"],
            "files": [],
            "retire_policy": "report-only: no merged copy of this folder's code exists, so it is never auto-disabled or deleted",
        }

    return {
        "_comment": (
            "PROJECT INVISIBLE - SamplerScheduler. Folders listed here were "
            "superseded by this engine. Nothing is deleted until a Generate has "
            "succeeded with a merged sampler AND every hash below verifies. "
            "Forge core files are never listed and never touched."
        ),
        "generated_at": now_iso(),
        "never_touch": list(NEVER_TOUCH),
        "folders": folders,
    }


def write_manifest() -> dict[str, Any]:
    existing = read_json(MANIFEST_PATH, {}) or {}
    if existing.get("consolidated_at"):
        return existing
    manifest = build_manifest()
    write_json(MANIFEST_PATH, manifest)
    return manifest


def load_manifest() -> dict[str, Any]:
    return read_json(MANIFEST_PATH, None) or build_manifest()


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
def verify_folder(folder: str, manifest: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    """Is it safe to retire ``folder``? Returns (ok, reasons_if_not)."""
    manifest = manifest or load_manifest()
    entry = (manifest.get("folders") or {}).get(folder)
    problems: list[str] = []

    if entry is None:
        return False, [f"{folder} is not in the manifest"]
    if folder in NEVER_TOUCH:
        return False, [f"{folder} is on the never-touch list"]
    if not entry.get("files"):
        return False, [f"no merged copy of {folder}'s code exists in lib/wrappers"]

    for rec in entry["files"]:
        our_abs = os.path.join(WRAPPERS_DIR, rec["merged_to"].split("lib/wrappers/", 1)[-1])
        if not os.path.isfile(our_abs):
            problems.append(f"merged copy missing: {rec['merged_to']}")
            continue
        if sha256(our_abs) != rec.get("merged_sha256"):
            problems.append(f"merged copy changed since merge: {rec['merged_to']}")

        src_abs = os.path.join(entry["path"], rec["source"])
        if os.path.isfile(src_abs) and sha256(src_abs) != rec.get("source_sha256"):
            problems.append(f"original changed since merge: {folder}/{rec['source']}")

    return (not problems), problems
