"""Fetch checksum-pinned upstream source capsules; licenses remain in archives."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = json.loads((ROOT / "configs/official_baseline.json").read_text())
    folder = ROOT / "assets/upstream"
    folder.mkdir(parents=True, exist_ok=True)
    for key in ("microduck_rl", "bam"):
        row = cfg[key]
        path = folder / row["archive"]
        if not path.exists():
            repository = row["repository"].removeprefix("https://github.com/")
            urllib.request.urlretrieve(
                f"https://codeload.github.com/{repository}/tar.gz/{row['commit']}", path
            )
        if hashlib.sha256(path.read_bytes()).hexdigest() != cfg["sha256"][path.name]:
            raise ValueError(f"Upstream archive checksum mismatch: {path}")
    scratch = ROOT / ".cache/official-export"
    scratch.mkdir(parents=True, exist_ok=True)
    with tarfile.open(folder / "microduck_rl.tar.gz") as archive:
        archive.extractall(scratch, filter="data")
    project = scratch / ("microduck_rl-" + cfg["microduck_rl"]["commit"])
    subprocess.run(
        [
            "uv",
            "--cache-dir",
            str(ROOT / ".cache/uv"),
            "export",
            "--project",
            str(project),
            "--frozen",
            "--no-emit-project",
            "--no-emit-package",
            "better-actuator-models",
            "--no-annotate",
            "--no-header",
            "--output-file",
            str(folder / "requirements-official.txt"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    for name, sha in cfg["sha256"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Dependency export mismatch: {name}; use uv {cfg['uv_version']}")
    print("Verified official capsules and dependency lock export")


if __name__ == "__main__":
    main()
