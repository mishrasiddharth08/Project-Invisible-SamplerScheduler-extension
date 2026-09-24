"""Proven sampler / scheduler implementations, merged from the old extensions.

Nothing in this package registers anything.  Each module only *defines* the
maths; ``lib/register.py`` owns every call into Forge's registries.  That split
is what makes registration idempotent and reversible.

Provenance of each file is recorded in ``MANIFEST_OLD_FOLDERS.json`` together
with a SHA-256 of the copy that lives here, so the merge step can prove a
hash-verified copy exists before it retires the original folder.
"""
