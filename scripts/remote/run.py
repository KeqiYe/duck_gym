"""Sync, verify, build, and launch a persistent GPU task in the approved workspace."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HOSTS = [
    ("master172", "/public/yekq6Data/codex/duck_gym"),
    ("delltower", "/home/yekeqi/Documents/HDD1/codex/duck_gym"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--setup", action="store_true")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    for host, workspace in HOSTS:
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
        raise SystemExit("Neither configured host is reachable")
    print(check.stdout, flush=True)
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
        for p in (ROOT / "assets/microduck").rglob("*")
        if p.is_file() and not p.name.startswith("._")
    ]
    manifest = dict(
        host=host,
        workspace=workspace,
        gpu=args.gpu,
        command=args.command,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        sha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
    )
    (local / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
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
        ["scp", str(local / "manifest.json"), host + ":" + remote_run + "/manifest.json"],
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
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="{workspace}/venv/bin:$CUDA_HOME/bin:$PATH"
export DUCK_CUDA_BUILD={shlex.quote(workspace+'/build/cuda')}
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi -i {args.gpu} --query-gpu=compute_cap --format=csv,noheader)"
export MAX_JOBS=2
export PYTHONUNBUFFERED=1
export DUCK_RUN_DIR={shlex.quote(remote_run)}
trap 'rc=$?; echo "$rc" > "$DUCK_RUN_DIR/exit_code"' EXIT
hostname
uname -a
nvidia-smi
"$CUDA_HOME/bin/nvcc" --version
"""
    if args.setup:
        script += f"bash scripts/remote/provision.sh {shlex.quote(workspace)}\n"
    else:
        script += 'python -m pip check\npython -m pip freeze > "$DUCK_RUN_DIR/dependencies.txt"\npython scripts/build_cuda.py\n'
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
