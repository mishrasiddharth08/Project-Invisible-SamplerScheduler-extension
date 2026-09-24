# PROJECT INVISIBLE — SamplerScheduler

Version **1.1.0** · Forge Neo · reviewed 11 September 2026.

One extension consolidating ExtraSchedulers, Neo Extra Samplers and the Forge
RES4LYF port. It adds choices to Forge's existing sampler and scheduler menus.
No generation tab, accordion, button or always-visible Script component.

## Included

This Forge build registers **46 additional samplers and 8 additional schedulers**.
Built-in names win; counts can change when Forge gains new implementations.

- Extra Samplers: Gradient Estimation and CFG++, SEEDS 2/3, SA Solver/PECE,
  DPM++ SDE CFG++, EXP Heun 2 x0/SDE, Res Multistep variants, Euler A2.
- ExtraSchedulers: Refined Exponential Solver, DPM++ 4M SDE,
  Euler Dy CFG++ and Euler SMEA Dy CFG++.
- RES4LYF: 16 RES/DEIS variants, including ODE versions.
- Reviewed ComfyUI ports: iPNDM/V, DEIS, DDPM, Heun++2, DPM2 a,
  DPM fast/adaptive, CFG++ UD10 AB and DPM++ 2S a CFG++.
- Local Forge implementations: DPM++ 2S a, DPM++ 2M SDE Heun, UniPC BH2.
- Schedulers: Cosine, CosineExponential blend, Phi, Laplace, Karras Dynamic,
  Custom, Tan and Beta57.

## Install / update

Place this folder under `extensions/project-invisible-samplerscheduler`, then
fully restart Forge. Keep this one extension; the three source extensions are
superseded. The installer checks syntax and dependencies and installs nothing.
No Forge core file is changed. The extension never deletes other folders at startup
or in response to saving an image. Consolidation is an explicit deployment action.

## Settings

Custom sigma expressions live in Settings → PROJECT INVISIBLE — SamplerScheduler.
Default: `m + (M-m)*(1-x)**3`. Expressions allow arithmetic and named math functions;
attribute access and arbitrary Python constructs are rejected. Invalid or ascending
curves fall back to a linear ramp. Literal lists are also supported.

Copy `config.example.json` to `config.json` if desired. Denylists accept labels,
function names or sampler aliases. User configuration is kept across updates.

## ComfyUI coverage

The pinned upstream audit covers 45 sampler names and all nine core schedulers.
39 sampler names resolve locally; four GPU-noise variants are intentionally
represented by their generic equivalents, while DDIM and UniPC use Forge's own
entries. GPU-noise variants are not bit-identical seed replacements. ComfyUI's
DDIM wrapper also has different inpainting semantics from Forge's DDIM.
No remaining distinct general KSampler algorithm is pending in this snapshot.
This is not a port of every custom node or video-specific sampling pipeline.

The optional weekly check reads public upstream name lists only. It writes
`pending.json`; it never executes downloaded code or registers entries from a
background thread. New algorithms require a reviewed update.

## Validation and limits

264 numerical checks passed using this machine's Forge modules, PyTorch 2.13.0,
Python 3.13 and RTX 5090: 46 samplers × CPU/CUDA × epsilon/flow inputs, plus
80 scheduler cases. Additional assertions cover duplicate registration, unload
and reload, upstream parsing, same-seed noise, inpainting mask restoration and
invalid custom expressions. Six host-free tests pass as well.

The numerical denoiser is synthetic. A full checkpoint image generation and
visual quality comparison were not performed. These checks do not establish
that every sampler/scheduler/model combination is useful. DPM fast/adaptive use
their own step sequence; DPM adaptive also chooses its own step count.

## Organization

- `scripts/engine.py`: startup, Settings callback, inventory and unload.
- `lib/register.py`: candidate catalog, stock precedence and object ownership.
- `lib/wrappers/`: sampler math, scheduler math and the RES4LYF bridge.
- `lib/comfy_sync.py`: read-only upstream audit.
- `lib/merge.py`: source provenance and hash verification helpers.
- `docs/UPSTREAM.md`, `docs/COMMUNITY_REFERENCES.md`: sources and review notes.
- `docs/coverage.json`, `docs/validation.json`: machine-readable audit results.
- `tests/`: host-free checks and the optional Forge numerical smoke test.

Run `python tests/test_no_host.py` anywhere. For the numerical suite, set
`PI_FORGE_ROOT` to the Forge checkout and `PI_TEST_OUTPUT` to a writable JSON
report path, then run `tests/forge_smoke.py` with Forge's Python environment.
The suite loads Forge libraries without loading checkpoints or starting the UI.

## Attribution

Original licences and credits are retained in `LICENSE`, `NOTICE`, and
`licenses/`. The RES4LYF licence restriction remains applicable to its components.
See those files before redistributing or offering hosted services.
