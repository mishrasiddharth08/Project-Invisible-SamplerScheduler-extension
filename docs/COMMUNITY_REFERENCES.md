# Reddit references reviewed

Community comparisons are anecdotal and depend on checkpoint, seed, step count,
guidance and scheduler. They informed what to preserve and test, not proof of
correctness or a universal default.

- [Flux-dev sampler/scheduler comparison](https://www.reddit.com/r/comfyui/comments/1elq2rk/fluxdev_samplers_and_schedulers_comparison/): compares many pairings; useful motivation for keeping sampler and scheduler independent.
- [RES Multistep discussion](https://www.reddit.com/r/StableDiffusion/comments/1i3zg7t/): examples with Flux/Beta and comparisons to DPM++; supports retaining the RES family, without asserting superiority.
- [HiDream sampler/scheduler compatibility tests](https://www.reddit.com/r/StableDiffusion/comments/1k5h18j/): model-specific compatibility examples; not transferable guarantees for every model.
- [HiDream / RES4LYF discussion](https://www.reddit.com/r/StableDiffusion/comments/1kgfolu/): community interest in RES 2S and RES 3M; those remain included.

No code was copied from Reddit. Primary algorithm sources are in UPSTREAM.md.
