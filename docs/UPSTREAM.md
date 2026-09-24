# Upstream audit — 2026-09-11

ComfyUI revision: `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`.

- [Sampler registry](https://github.com/Comfy-Org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/samplers.py)
- [Sampling algorithms](https://github.com/Comfy-Org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/k_diffusion/sampling.py)
- [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic)
- [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF)

`comfy_latest.py` copies `_sample_cfgpp_history`, `sample_cfgpp_ud10_ab`,
`sample_dpmpp_2s_ancestral_cfg_pp` and the multistep integration helper from this
pinned revision. Changes: Forge predictor lookup, Forge post-CFG hook registration,
local seeded noise, no-gradient context. The UD10 adapter explicitly requests the
unconditional branch because Forge's hook arguments differ at CFG=1. UniPC BH2
calls Forge's implementation with `variant="bh2"`.

The older DPM2 ancestral RF path contained one missed ComfyUI model-patcher lookup;
it now uses the same Forge predictor adapter as the other ports.
Dynamic CFG++ samplers now capture unconditional output through Forge's current
hook, without leaving model-option mutations behind. Their temporary inpainting
mask/latent resizing is forwarded to the actual denoiser and restored on exit.

Scheduler corrections: one-step Cosine/Phi/blend start at sigma_max;
Karras Dynamic recomputes both bounds at the varying rho; Beta57 retains the
requested number of lookup positions; Custom uses restricted syntax and checks
finite, positive, descending results. These are documented local corrections,
not claims of byte-identical upstream scheduler implementations.

Existing bundled implementations were retained unless a concrete incompatibility
was found. A current upstream name audit is not a claim that every old sampler's
code is identical to the latest ComfyUI revision. No entire ComfyUI dependency
or custom-node pipeline is installed. `coverage.json` records all names and
exceptions; `validation.json` records the synthetic numerical checks.
