"""Read-only dependency and syntax check. Never installs or deletes anything."""
from pathlib import Path
import importlib.util


def main():
    root = Path(__file__).resolve().parent
    missing = [name for name in ("torch", "numpy", "scipy", "torchsde", "tqdm")
               if importlib.util.find_spec(name) is None]
    if missing:
        print("[Invisible-SS] Missing Forge dependencies: " + ", ".join(missing))
    files = [*root.joinpath("lib").rglob("*.py"), *root.joinpath("scripts").glob("*.py")]
    for path in files:
        compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
    print(f"[Invisible-SS] Syntax checked: {len(files)} files. No packages installed.")


main()
