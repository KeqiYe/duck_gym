"""Build the CUDA extension on the selected remote host, never on macOS."""

from pathlib import Path
import os
import platform


def build():
    if platform.system() != "Linux":
        raise RuntimeError("CUDA builds run on the configured SSH host, not the local Mac")
    import torch
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[1]
    build_dir = Path(os.environ["DUCK_CUDA_BUILD"]).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
    os.environ.setdefault("MAX_JOBS", "2")
    return load(
        name="_duck_cuda",
        sources=[str(root / "src/cuda/extension.cu"), str(root / "src/solver.cpp")],
        extra_include_paths=[
            str(root / "include"),
            str(build_dir.parents[1] / "cache/python-dev/usr/include/python3.12"),
            str(build_dir.parents[1] / "cache/python-dev/usr/include"),
        ],
        build_directory=str(build_dir),
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--fmad=false"],
        verbose=True,
    )


if __name__ == "__main__":
    extension = build()
    print(extension.__file__)
