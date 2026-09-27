"""Restore pinned MicroDuck collision masks and export native DUCK_MODEL 2."""

import bootstrap
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import mujoco
from audit_bam_contacts import build_audit_model
from model_io import export_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new model directory")
    shutil.copytree(args.source, args.output)
    model, pairs = build_audit_model(
        args.output, explicit_pairs=False, xml_path=args.output / "self-contact.xml"
    )
    data = mujoco.MjData(model)
    cfg = json.loads((args.output / "config.json").read_text())
    data.qpos[:] = cfg["default_qpos"]
    export_model(model, data, args.output / "model.duck", self_contacts=True)
    cfg.update(
        scope="Nominal BAM model with original ground and self collision masks; native convex MPR/AVBD",
        omissions=["sensor observations", "per-env domain randomization"],
        self_contact_pairs=pairs,
        self_contact_model_source=str(args.source),
        source_model_sha256=hashlib.sha256((args.source / "model.duck").read_bytes()).hexdigest(),
        native_model_format=2,
    )
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(args.output),
                pairs=pairs,
                sha256=hashlib.sha256((args.output / "model.duck").read_bytes()).hexdigest(),
            )
        )
    )


if __name__ == "__main__":
    main()
