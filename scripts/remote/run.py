"""Sync, verify, build, and launch a persistent GPU task in the approved workspace."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
HOSTS = [
    ("master172", "/public/yekq6Data/codex/duck_gym"),
    ("delltower", "/home/yekeqi/Documents/HDD1/codex/duck_gym"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", choices=["auto"] + [host for host, _ in HOSTS], default="auto",
                   help="Default: preferred host then fallback; select a host explicitly for a requested GPU model")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--cuda-home", default="/usr/local/cuda-12.8",
                   help="CUDA Toolkit root on the selected host (must contain bin/nvcc)")
    p.add_argument("--setup", action="store_true")
    p.add_argument("--native-float-math", action="store_true", help="Experimental FP32 native math; independent accuracy validation required")
    p.add_argument("--unroll-small-matrices", action="store_true", help="Experimental fixed-size matrix loop expansion; compare accuracy and register use")
    p.add_argument("--python-env", default="venv", help="Workspace environment directory name")
    p.add_argument("--build-config", default="cuda", help="Isolated build directory name")
    p.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="Explicitly allow sharing a GPU with running compute processes",
    )
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    for name in (args.python_env, args.build_config):
        if not name or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in name
        ):
            p.error("Environment/build names must use letters, digits, underscore or hyphen")
    selected_hosts = HOSTS if args.host == "auto" else [(host, workspace) for host, workspace in HOSTS if host == args.host]
    for host, workspace in selected_hosts:
        check = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                host,
                "hostname; nvidia-smi; test -d " + shlex.quote(str(Path(workspace).parent)),
            ],
            text=True,
            capture_output=True,
        )
        if check.returncode == 0:
            break
        print(f"{host} unavailable: {check.stderr}", flush=True)
    else:
        raise SystemExit("No selected host is reachable: " + ", ".join(host for host, _ in selected_hosts))
    print(check.stdout, flush=True)
    occupied = subprocess.run(
        [
            "ssh",
            host,
            f"nvidia-smi -i {args.gpu} --query-compute-apps=pid,process_name --format=csv,noheader",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if occupied and not args.allow_busy_gpu:
        raise SystemExit(
            f"GPU {args.gpu} is occupied; select a free GPU or explicitly use --allow-busy-gpu.\n{occupied}"
        )
    run_id = datetime.now().strftime("gpu-%Y%m%d-%H%M%S-%f")
    local = ROOT / "runs" / run_id
    local.mkdir(parents=True)
    files = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    files = [f for f in files if f and (ROOT / f).is_file()]
    files += [
        str(p.relative_to(ROOT))
        for folder in ("assets/microduck", "assets/upstream")
        for p in (ROOT / folder).rglob("*")
        if p.is_file() and not p.name.startswith("._")
    ]
    manifest = dict(
        host=host,
        host_selection=args.host,
        workspace=workspace,
        gpu=args.gpu,
        cuda_home=args.cuda_home,
        python_environment=args.python_env,
        build_configuration=args.build_config,
        native_float_math=args.native_float_math,
        unroll_small_matrices=args.unroll_small_matrices,
        command=args.command,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        sha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
    )
    (local / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Hashes alone cannot reconstruct uncommitted/untracked experiments. Keep
    # the actual source/config text; fixed-version assets remain hash-addressed.
    snapshot = local / "source_snapshot.tar.gz"
    with tarfile.open(snapshot, "w:gz") as archive:
        for name in files:
            if name.startswith(("assets/microduck/", "assets/upstream/")):
                continue
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != manifest["sha256"][name]:
                raise RuntimeError(f"Source changed during snapshot: {name}; retry synchronization")
            archive.add(ROOT / name, arcname=name, recursive=False)
    subprocess.run(
        [
            "ssh",
            host,
            f'mkdir -p {shlex.quote(workspace+"/source")} {shlex.quote(workspace+"/runs/"+run_id)}',
        ],
        check=True,
    )
    subprocess.run(
        [
            "rsync",
            "-az",
            "--delete",
            "--exclude=.git",
            "--exclude=.venv",
            "--exclude=.cache",
            "--exclude=build",
            "--exclude=runs",
            "--exclude=__pycache__",
            "--exclude=.DS_Store",
            "--exclude=._*",
            str(ROOT) + "/",
            host + ":" + workspace + "/source/",
        ],
        check=True,
    )
    remote_run = workspace + "/runs/" + run_id
    subprocess.run(
        ["scp", str(local / "manifest.json"), str(snapshot), host + ":" + remote_run + "/"],
        check=True,
    )
    verify = (
        "import hashlib,json,pathlib; m=json.load(open("
        + repr(remote_run + "/manifest.json")
        + ")); root=pathlib.Path("
        + repr(workspace + "/source")
        + '); bad=[p for p,h in m["sha256"].items() if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]; assert not bad,bad;print("Verified",len(m["sha256"]),"source/asset files")'
    )
    subprocess.run(["ssh", host, "python3 -c " + shlex.quote(verify)], check=True)
    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd and not args.setup:
        p.error("Provide a remote command or --setup")
    script = f"""#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(workspace+'/source')}
export CUDA_VISIBLE_DEVICES={args.gpu}
export CUDA_HOME={shlex.quote(args.cuda_home)}
export PATH="{workspace}/{args.python_env}/bin:{workspace}/venv/bin:$CUDA_HOME/bin:$PATH"
export DUCK_CUDA_BUILD={shlex.quote(workspace+'/build/'+args.build_config)}
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi -i {args.gpu} --query-gpu=compute_cap --format=csv,noheader)"
export MAX_JOBS=2
export DUCK_NATIVE_FLOAT_MATH={int(args.native_float_math)}
export DUCK_UNROLL_SMALL_MATRICES={int(args.unroll_small_matrices)}
export PYTHONUNBUFFERED=1
export DUCK_RUN_DIR={shlex.quote(remote_run)}
trap 'rc=$?; echo "$rc" > "$DUCK_RUN_DIR/exit_code"' EXIT
hostname
uname -a
nvidia-smi
"$CUDA_HOME/bin/nvcc" --version
"${{CXX:-c++}}" --version
python --version
if command -v ninja > /dev/null 2>&1; then ninja --version; fi
"""
    if args.setup:
        script += f"bash scripts/remote/provision.sh {shlex.quote(workspace)}\n"
    else:
        script += f"""if python -m pip --version >/dev/null 2>&1; then
  python -m pip check
  python -m pip freeze > "$DUCK_RUN_DIR/dependencies.txt"
else
  "{workspace}/tools/bin/uv" pip check --python "{workspace}/{args.python_env}/bin/python"
  "{workspace}/tools/bin/uv" pip freeze --python "{workspace}/{args.python_env}/bin/python" > "$DUCK_RUN_DIR/dependencies.txt"
fi
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print("PyTorch", torch.__version__, "CUDA runtime", torch.version.cuda, "CUDA_HOME", CUDA_HOME)'
python scripts/build_cuda.py
"""
        script += shlex.join(cmd) + "\n"
    (local / "run.sh").write_text(script)
    subprocess.run(["scp", str(local / "run.sh"), host + ":" + remote_run + "/run.sh"], check=True)
    launch = f'nohup bash {shlex.quote(remote_run+"/run.sh")} > {shlex.quote(remote_run+"/run.log")} 2>&1 < /dev/null & echo $! > {shlex.quote(remote_run+"/pid")}; cat {shlex.quote(remote_run+"/pid")}'
    result = subprocess.check_output(["ssh", host, launch], text=True)
    print(
        json.dumps(
            dict(host=host, run_dir=remote_run, pid=result.strip(), local=str(local)), indent=2
        )
    )


if __name__ == "__main__":
    main()
