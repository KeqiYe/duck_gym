#!/usr/bin/env bash
set -euo pipefail
workspace="${1:?Pass the configured remote duck_gym workspace}"
variant="${2:-benchmark}"
bench="$workspace/venv-$variant"
if [[ ! -x "$bench/bin/python" ]]; then
  "$workspace/venv/bin/python" -m venv --without-pip "$bench"
fi
# Keep PyTorch/CUDA wheels in the existing workspace; benchmark packages take
# precedence in this venv and installation never upgrades the training venv.
site_dir="$($bench/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
training_site="$($workspace/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf '%s\n' "$training_site" > "$site_dir/training_torch.pth"
export PIP_CACHE_DIR="$workspace/cache/pip"
"$workspace/venv/bin/python" -m pip --python "$bench/bin/python" install \
  --timeout 120 --retries 5 -r "$workspace/source/requirements-$variant.txt"
"$bench/bin/python" -m pip check
"$bench/bin/python" -m pip freeze > "$DUCK_RUN_DIR/benchmark-dependencies.txt"
WARP_CACHE_PATH="$workspace/cache/warp-benchmark" "$bench/bin/python" -c 'import os,mujoco,mujoco_warp,warp,torch; print("MuJoCo",mujoco.__version__,"MJWarp",mujoco_warp.__version__,"Warp",warp.__version__,"Torch",torch.__version__); warp.config.kernel_cache_dir=os.environ["WARP_CACHE_PATH"]; warp.init(); print(warp.get_cuda_devices())'
