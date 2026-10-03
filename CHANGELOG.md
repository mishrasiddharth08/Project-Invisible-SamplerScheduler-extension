# Changelog

## 1.2.0 — 2026-10-04

- Integrated the DPM++ 2M Sharp family from envy-ai/ComfyUI-DPMpp-2M-Sharp:
  DPM++ 2M Sharp, DPM++ 2M SDE GPU Sharp, SEEDS 2 Sharp, RES 2S/2M NC and
  their Sharp variants (7 new samplers, plus the `sharpness` extra param).
- Host-API adaptations documented in lib/wrappers/dpmpp_sharp.py; provenance
  recorded in NOTICE.
- README updated with the new entries and tuning guidance.

## 1.1.0 — 2026-09-11

- Consolidated the three source extensions under one invisible Forge extension.
- Added reviewed CFG++ UD10 AB, DPM++ 2S a CFG++ and UniPC BH2 entries.
- Made DPM++ 2S a registration available on every startup, including offline.
- Fixed Forge guidance-hook compatibility in dynamic CFG++ and the DPM2 RF path.
- Fixed live ownership, unload/reload, hidden-scheduler registration and alias denylists.
- Replaced regex-only ComfyUI parsing with non-executing AST list/handler parsing.
- Made the weekly update check report-only and removed automatic deletion hooks.
- Corrected scheduler bounds and constrained custom expression evaluation.
- Added numerical regressions, pinned upstream provenance and Reddit references.
