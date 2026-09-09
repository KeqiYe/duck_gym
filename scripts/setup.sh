#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
command -v cmake >/dev/null || { echo 'Install CMake first.' >&2; exit 1; }
command -v c++ >/dev/null || { echo 'Install a C++17 compiler first.' >&2; exit 1; }
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 11), "Python >= 3.11 required"'
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install -r requirements.txt --timeout 120 --retries 5
.venv/bin/python scripts/fetch_assets.py
cmake -S . -B build/cpu-release -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu-release --parallel

ctest --test-dir build/cpu-release --output-on-failure
.venv/bin/python -m pip check
