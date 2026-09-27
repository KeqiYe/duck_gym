"""Export a native rsl-rl 5 checkpoint without constructing a physics environment."""

import bootstrap
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--training-run", type=Path, required=True)
    args = p.parse_args()
    out = Path(os.environ["DUCK_RUN_DIR"])
    record = json.loads((args.training_run / "native_training.json").read_text())
    cfg = deepcopy(record["runner"]["actor"])
    assert cfg.pop("class_name") == "MLPModel"
    torch.manual_seed(71)
    sample = torch.randn(64, 61) * 0.01
    obs = TensorDict({"actor": sample}, batch_size=[64])
    actor = MLPModel(obs, {"actor": ["actor"]}, "actor", 14, **cfg).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    deploy = actor.as_onnx(verbose=False).eval()
    model = json.loads((Path(record["arguments"]["model_dir"]) / "config.json").read_text())
    head_ids = [i for i,n in enumerate(model['joint_names']) if 'head' in n or 'neck' in n]
    with torch.no_grad():
        expected = actor(obs)
        dummy = deploy.get_dummy_inputs()
        if record['task'].get('neutral_head',False):
            from duck_gym.policy_wrappers import NeutralHeadPolicy
            deploy = NeutralHeadPolicy(deploy,head_ids).eval()
            expected[:,head_ids] = 0
        traced = torch.jit.trace(deploy, dummy)
        actual = traced(sample)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        traced.save(str(out / "policy.pt"))
    model = json.loads((Path(record["arguments"]["model_dir"]) / "config.json").read_text())
    metadata = dict(
        joint_names=[n.split("/")[-1] for n in model["joint_names"]],
        action_scale=1.0,
        clip_actions=record["runner"].get("clip_actions"),
        normalization="Embedded upstream deployment wrapper",
        training_run=str(args.training_run),
        heading_hold=record["task"].get("heading_hold", False),
        checkpoint=str(args.checkpoint),
        iteration=checkpoint["iter"],
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        actual_upstream_task=False,
        neutral_head=record['task'].get('neutral_head',False),
        export_max_error=float((actual - expected).abs().max()),
    )
    (out / "policy.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.savez(out / "policy-check.npz", observations=sample.numpy(), actions=expected.numpy())
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
