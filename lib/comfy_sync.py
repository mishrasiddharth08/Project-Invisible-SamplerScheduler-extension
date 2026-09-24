"""Read-only ComfyUI name audit. Never executes downloads or changes registries."""
from __future__ import annotations

import ast
import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from . import PENDING_PATH, debug, log, now_iso, warn, write_json

RAW_SAMPLERS_PY = "https://raw.githubusercontent.com/Comfy-Org/ComfyUI/master/comfy/samplers.py"
API_COMMITS = "https://api.github.com/repos/Comfy-Org/ComfyUI/commits?path=comfy/samplers.py&per_page=1"

_UA = {"User-Agent": "PROJECT-INVISIBLE-SamplerScheduler/1.0 (+forge-neo extension)"}

#: ComfyUI names that must never be auto-registered here.
_NEVER_SYNC = {
    # Forge Neo selects the RF trunk internally for rectified-flow models;
    # a second RF twin in the dropdown would be a trap, not a feature.
    "euler_ancestral_RF",
    "dpmpp_2s_ancestral_RF",
    # Meta-entries, not samplers.
    "ddim",
    "uni_pc",
}


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------
def _get(url: str, timeout: int) -> str | None:
    req = urllib.request.Request(url, headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        debug(f"comfy sync HTTP {exc.code} for {url}")
    except Exception as exc:
        debug(f"comfy sync network error for {url}: {exc}")
    return None


def _remote_sha(timeout: int) -> str | None:
    body = _get(API_COMMITS, timeout)
    if not body:
        return None
    try:
        data = json.loads(body)
        if isinstance(data, list) and data:
            return str(data[0].get("sha"))[:12]
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
# Parse (text only - nothing is executed)
# --------------------------------------------------------------------------
def _parse_list(source: str, varname: str) -> list[str]:
    """Extract a top-level ``VARNAME = [...]`` list of string literals."""
    m = re.search(rf"^{re.escape(varname)}\s*=\s*\[(.*?)\]", source, re.S | re.M)
    if not m:
        return []
    return re.findall(r"[\"']([A-Za-z0-9_+\-]+)[\"']", m.group(1))


def parse_name_lists(source: str) -> dict[str, list[str]]:
    """Read literal lists, list concatenation and handler keys without execution."""
    values = {}
    def read(node):
        if isinstance(node, (ast.List, ast.Tuple)):
            return [n.value for n in node.elts if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        if isinstance(node, ast.Dict):
            return [n.value for n in node.keys if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        if isinstance(node, ast.Name):
            return values.get(node.id, [])
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return read(node.left) + read(node.right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "list" and len(node.args) == 1:
            return read(node.args[0])
        return []
    try:
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        values[target.id] = read(node.value)
    except SyntaxError:
        return {key: [] for key in ("KSAMPLER_NAMES", "SAMPLER_NAMES", "SCHEDULER_NAMES")}
    return {key: values.get(key, []) for key in ("KSAMPLER_NAMES", "SAMPLER_NAMES", "SCHEDULER_NAMES")}


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
#: Token -> dropdown spelling, chosen to match how Forge Neo already writes
#: its own sampler labels ("DPM++ 2M SDE", "Euler a", "ER SDE").
_TOKENS = {
    "dpmpp": "DPM++",
    "cfgpp": "CFG++",
    "ancestral": "a",
    "sde": "SDE",
    "ode": "ODE",
    "gpu": "GPU",
    "sa": "SA",
    "er": "ER",
    "lcm": "LCM",
    "lms": "LMS",
    "ddpm": "DDPM",
    "ddim": "DDIM",
    "deis": "DEIS",
    "ipndm": "iPNDM",
    "res": "Res",
    "rf": "RF",
    "a": "a",
    "x0": "x0",
}


def prettify(name: str) -> str:
    """Turn ``dpmpp_2m_sde`` into ``DPM++ 2M SDE``, matching Forge's own style."""
    words = []
    for part in name.replace("_cfg_pp", "_cfgpp").split("_"):
        low = part.lower()
        if low in _TOKENS:
            words.append(_TOKENS[low])
        elif re.fullmatch(r"\d+[a-z]?", low):
            words.append(low.upper())
        else:
            words.append(low.capitalize())
    return " ".join(words)


def _claimed_identities() -> set[str]:
    """Every function identity already claimed by a registered sampler.

    A prettified upstream label will rarely equal Forge's own wording for the
    same sampler ("Euler ANCESTRAL" vs "Euler a"), so comparing labels is not
    enough to spot a duplicate.  What actually identifies a sampler is the
    k-diffusion function behind it, so this collects, for every registered
    entry: its aliases, those aliases with a ``k_``/``sample_`` prefix removed,
    and - where the constructor closure exposes it - the resolved function name.
    """
    from .detect import host

    ids: set[str] = set()

    def _add(token: str) -> None:
        low = token.lower()
        ids.add(low)
        for prefix in ("k_", "sample_"):
            if low.startswith(prefix):
                ids.add(low[len(prefix) :])

    sd_samplers = host().get("sd_samplers")
    if sd_samplers is None:
        return ids
    for s in getattr(sd_samplers, "all_samplers", []):
        _add(s.name.replace(" ", "_"))
        for a in s.aliases or []:
            _add(a)
        # SamplerData.constructor is ``lambda model, _f=<func or name>: ...``;
        # the default holds the identity Forge will actually sample with.
        try:
            for default in getattr(s.constructor, "__defaults__", None) or ():
                if isinstance(default, str):
                    _add(default)
                elif callable(default):
                    _add(getattr(default, "__name__", ""))
        except Exception:
            pass
    ids.discard("")
    return ids


def classify(remote: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    """Split upstream names into already-registered / resolvable / needs-port."""
    from .detect import current_sampler_names, host, k_diffusion_has

    known_labels = {n.lower() for n in current_sampler_names()}
    claimed = _claimed_identities()
    remote_samplers = remote.get("SAMPLER_NAMES") or remote.get("KSAMPLER_NAMES") or []

    already: list[dict[str, Any]] = []
    resolvable: list[dict[str, Any]] = []
    needs_port: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    generic_present = {n for n in remote_samplers if not n.endswith("_gpu")}

    for name in remote_samplers:
        rec = {"name": name, "label": prettify(name), "func": f"sample_{name}"}

        if name in _NEVER_SYNC:
            rec["reason"] = "excluded: Forge Neo handles this trunk internally"
            skipped.append(rec)
            continue

        # Skip GPU twins when the generic variant covers it (spec: speed/memory).
        if name.endswith("_gpu") and name[: -len("_gpu")] in generic_present:
            rec["reason"] = "generic (CPU-noise) twin is sufficient; GPU twin skipped"
            skipped.append(rec)
            continue

        # Identity match beats label match: Forge spells ``euler_ancestral`` as
        # "Euler a", so only the function identity reveals the duplicate.
        if name.lower() in claimed or rec["func"].lower() in claimed or rec["label"].lower() in known_labels:
            rec["reason"] = "already registered in this Forge tree (same k-diffusion function)"
            already.append(rec)
            continue

        if k_diffusion_has(rec["func"]):
            rec["reason"] = "implemented by Forge's bundled k_diffusion; trivial alias"
            resolvable.append(rec)
        else:
            rec["reason"] = "no local implementation; needs a reviewed Forge-Neo port before it can be registered"
            needs_port.append(rec)

    # Schedulers: report only. A scheduler name without its sigma function is
    # not portable, and inventing the curve here is out of scope.
    from .detect import current_scheduler_labels, current_scheduler_names

    known_sched = {n.lower() for n in current_scheduler_names()} | {n.lower() for n in current_scheduler_labels()}
    if "ddim" in known_sched:
        known_sched.add("ddim_uniform")
    sched_new = [
        {"name": n, "reason": "not present locally; needs a reviewed sigma-schedule port"}
        for n in remote.get("SCHEDULER_NAMES", [])
        if n.lower() not in known_sched
    ]

    _ = host  # keep the import meaningful for readers of the call graph
    return {
        "already_registered": already,
        "resolvable": resolvable,
        "needs_port": needs_port,
        "skipped": skipped,
        "schedulers_needing_port": sched_new,
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def should_check(cfg: dict[str, Any]) -> bool:
    sync = cfg.get("comfy_sync") or {}
    if not sync.get("enabled", True):
        return False
    last = sync.get("last_check")
    if not last:
        return True
    try:
        last_ts = time.mktime(time.strptime(last, "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return True
    return (time.time() - last_ts) >= float(sync.get("interval_hours", 168)) * 3600.0


def run(cfg: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    """Perform one sync pass. Always returns a result dict; never raises."""
    sync = cfg.setdefault("comfy_sync", {})
    result: dict[str, Any] = {"ran": False, "reason": "", "registered": [], "pending": 0}

    if not force and not should_check(cfg):
        result["reason"] = "not due yet"
        return result

    timeout = int(sync.get("network_timeout_seconds", 10))
    source = _get(RAW_SAMPLERS_PY, timeout)
    if not source:
        result["reason"] = "ComfyUI master unreachable (offline or rate-limited); skipped"
        debug(result["reason"])
        return result

    remote = parse_name_lists(source)
    if not any(remote.values()):
        result["reason"] = "upstream file parsed to zero names; layout may have changed - skipped"
        warn(result["reason"])
        return result

    buckets = classify(remote)
    sha = _remote_sha(timeout)

    registered: list[str] = []  # Reports only; registries are finalized before UI construction.

    pending = {
        "_comment": (
            "Upstream ComfyUI names this engine did NOT register. Each needs a "
            "reviewed Forge-Neo port; nothing here is downloaded or executed."
        ),
        "checked_at": now_iso(),
        "comfy_sha": sha,
        "samplers_needing_port": buckets["needs_port"],
        "local_candidates_needing_review": buckets["resolvable"],
        "upstream_names": remote,
        "schedulers_needing_port": buckets["schedulers_needing_port"],
        "skipped_on_purpose": buckets["skipped"],
    }
    write_json(PENDING_PATH, pending)

    sync["last_check"] = now_iso()
    if sha:
        sync["last_comfy_sha"] = sha

    seen = cfg.setdefault("seen_names", {}).setdefault("samplers", [])
    for rec in buckets["already_registered"] + buckets["resolvable"]:
        if rec["name"] not in seen:
            seen.append(rec["name"])

    n_new = len(buckets["needs_port"]) + len(buckets["schedulers_needing_port"])
    log(
        f"comfy sync ({sha or 'sha unknown'}): {len(registered)} auto-registered, "
        f"{n_new} need a port (see pending.json), {len(buckets['skipped'])} skipped on purpose"
    )
    for rec in registered:
        log(f"  + {rec} -> auto-registered from Forge's own k_diffusion")

    result.update({"ran": True, "reason": "ok", "registered": registered, "pending": n_new, "sha": sha})
    return result
