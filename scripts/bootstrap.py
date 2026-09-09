"""Repository entry points share the same native extension and Python package."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "python"),
    os.environ.get("DUCK_NATIVE_PATH", str(ROOT / "build/cpu-release/python")),
]
