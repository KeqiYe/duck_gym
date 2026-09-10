#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -x .venv/bin/python ]] || bash scripts/setup.sh
.venv/bin/python -m pip install -r requirements-training.txt --timeout 120 --retries 5
cmake -S . -B build/cpu-release -DCMAKE_BUILD_TYPE=Release -DDUCK_BUILD_PYTHON=ON \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
cmake --build build/cpu-release --parallel
.venv/bin/python scripts/prepare_standing.py
.venv/bin/python scripts/prepare_gait.py
ctest --test-dir build/cpu-release --output-on-failure
.venv/bin/python tests/test_python.py
.venv/bin/python tests/test_tensor_env.py
.venv/bin/python tests/test_checkpoint.py
.venv/bin/python tests/test_evaluation.py
.venv/bin/python -m pip check
