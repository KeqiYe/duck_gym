"""Export a phase-conditioned skill checkpoint with embedded normalization."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--training-run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    record = json.loads((a.training_run / "skill_training.json").read_text())
    cfg = deepcopy(record["runner"]["actor"])
    assert cfg.pop("class_name", "MLPModel") == "MLPModel"
    # Early records were written after rsl-rl consumed constructor class names.
    # The pinned skill trainer uses the same scalar Gaussian as its baseline.
    cfg["distribution_cfg"].setdefault("class_name", "GaussianDistribution")
    torch.manual_seed(77)
    sample = torch.randn(32, record["observations"]["actor"]) * 0.01
    obs = TensorDict({"actor": sample}, batch_size=[32])
    actor = MLPModel(obs, {"actor": ["actor"]}, "actor", 14, **cfg).eval()
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    actor.load_state_dict(state["actor_state_dict"], strict=True)
    deploy = actor.as_onnx(verbose=False).eval()
    with torch.no_grad():
        traced = torch.jit.trace(deploy, deploy.get_dummy_inputs())
        torch.testing.assert_close(traced(sample), actor(obs), atol=2e-5, rtol=2e-5)
        traced.save(str(a.output / "policy.pt"))
    task = record["task"]
    if task.get("reference_trajectory"):
        task["reference_sha256"] = next(
            v
            for k, v in record["input_sha256"].items()
            if Path(k).name == Path(task["reference_trajectory"]).name
        )
    (a.output / "policy.json").write_text(
        json.dumps(
            dict(
                task=task,
                training_run=str(a.training_run),
                checkpoint=str(a.checkpoint),
                checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                actor_observations=record["observations"]["actor"],
                accepted=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
