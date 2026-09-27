#!/usr/bin/env bash
set -euo pipefail
workspace="${1:?Pass remote duck_gym workspace}"
export PIP_CACHE_DIR="$workspace/cache/pip"
export UV_CACHE_DIR="$workspace/cache/uv-official"
export UV_HTTP_TIMEOUT=120
export UV_LINK_MODE=copy
mkdir -p "$workspace/tools" "$workspace/build/upstream"
if [[ ! -x "$workspace/tools/bin/uv" ]]; then
  "$workspace/venv/bin/python" -m pip install --target "$workspace/tools" uv==0.11.7
fi
uv="$workspace/tools/bin/uv"
"$uv" --version
python3 - "$workspace" <<'PY'
import hashlib,json,sys,tarfile
from pathlib import Path
w=Path(sys.argv[1]); cfg=json.loads((w/'source/configs/official_baseline.json').read_text())
for name,h in cfg['sha256'].items():
    assert hashlib.sha256((w/'source/assets/upstream'/name).read_bytes()).hexdigest()==h,name
for key in ('microduck_rl','bam'):
    entry=cfg[key]; dest=w/'build/upstream'/entry['commit']
    if not dest.exists():
        dest.mkdir()
        with tarfile.open(w/'source/assets/upstream'/entry['archive']) as archive:
            archive.extractall(dest,filter='data')
print('Verified official source archives and dependency export')
PY
official="$workspace/build/upstream/1e79c29c97d8b38aee9eefde77a545860ba7658e/microduck_rl-1e79c29c97d8b38aee9eefde77a545860ba7658e"
bam="$workspace/build/upstream/62bd8ce12154340be97e06f7f41a0ca8f116d967/bam-62bd8ce12154340be97e06f7f41a0ca8f116d967"
envdir="$workspace/venv-official"
[[ -x "$envdir/bin/python" ]] || "$uv" venv --python /usr/bin/python3.12 "$envdir"
"$uv" pip sync --python "$envdir/bin/python" --require-hashes "$workspace/source/assets/upstream/requirements-official.txt"
"$uv" pip install --python "$envdir/bin/python" --no-deps "$bam" -e "$official"
"$uv" pip check --python "$envdir/bin/python"
"$uv" pip freeze --python "$envdir/bin/python" > "$DUCK_RUN_DIR/official-dependencies.txt"
"$envdir/bin/python" -c 'import torch,mujoco,warp,mjlab,bam; print("Official environment",torch.__version__,mujoco.__version__,warp.__version__,torch.cuda.get_device_name())'
"$envdir/bin/train" Mjlab-Velocity-Flat-MicroDuck --help > "$DUCK_RUN_DIR/train-help.txt"
