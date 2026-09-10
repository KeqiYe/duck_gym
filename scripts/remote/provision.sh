#!/usr/bin/env bash
set -euo pipefail
workspace="${1:?Pass remote duck_gym workspace root}"
mkdir -p "$workspace/build" "$workspace/runs" "$workspace/cache"
python3 -m venv --without-pip "$workspace/venv"
if ! "$workspace/venv/bin/python" -m pip --version >/dev/null 2>&1; then
  curl --fail --location --retry 5 https://bootstrap.pypa.io/get-pip.py -o "$workspace/cache/get-pip.py"
  "$workspace/venv/bin/python" "$workspace/cache/get-pip.py"
fi
export PIP_CACHE_DIR="$workspace/cache/pip"
"$workspace/venv/bin/python" -m pip install --timeout 120 --retries 5 torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$workspace/venv/bin/python" -m pip install --timeout 120 --retries 5 -r "$workspace/source/requirements-training.txt" ninja==1.11.1.4
"$workspace/venv/bin/python" -m pip check

# Ubuntu runtime may omit headers; extract the distro development package locally.
if [[ ! -f /usr/include/python3.12/Python.h ]]; then
  mkdir -p "$workspace/cache/python-dev"
  (cd "$workspace/cache/python-dev" && apt-get download libpython3.12-dev &&
    for package in ./*.deb; do dpkg-deb -x "$package" .; done)
fi
