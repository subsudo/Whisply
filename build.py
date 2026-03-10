from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent
DIST = ROOT / "dist"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_portable() -> None:
    required_assets = [
        ROOT / "assets" / "icon.ico",
        ROOT / "config.yaml",
        ROOT / "cuda_manifest.json",
    ]
    for asset in required_assets:
        if not asset.exists():
            sys.exit(f"Missing build asset: {asset}")

    run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--name",
            "Whisply",
            "--onefile",
            "--windowed",
            "--icon=assets/icon.ico",
            "--add-data",
            "config.yaml;.",
            "--add-data",
            "assets;assets",
            "--add-data",
            "cuda_manifest.json;.",
            "--hidden-import=faster_whisper",
            "--hidden-import=ctranslate2",
            "--hidden-import=av._core",
            "--collect-all",
            "ctranslate2",
            "--collect-all",
            "av",
            "main.py",
        ]
    )


def _find_iscc() -> str | None:
    candidates: list[str | None] = [
        shutil.which("ISCC.exe"),
        shutil.which("iscc"),
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if Path(candidate).exists():
            return candidate
    return None


def build_installer() -> None:
    iscc = _find_iscc()
    if not iscc:
        raise RuntimeError("Inno Setup Compiler not found (ISCC.exe). Install Inno Setup 6.")
    run([iscc, "installer.iss"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Whisply")
    parser.add_argument("--portable", action="store_true", help="Build portable exe only")
    parser.add_argument("--installer", action="store_true", help="Build installer only")
    parser.add_argument("--all", action="store_true", help="Build exe and installer")
    args = parser.parse_args()

    if not any([args.portable, args.installer, args.all]):
        args.all = True

    DIST.mkdir(exist_ok=True)

    try:
        if args.portable or args.all:
            build_portable()
        if args.installer or args.all:
            build_installer()
    except subprocess.CalledProcessError as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("Build complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
