"""Extend observation inputs without discarding learned policy or optimizer state."""

from pathlib import Path
import torch


def extend_observations(checkpoint, runner, output):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    current = runner.alg.policy.state_dict()
    changes = {}
    for key in ("actor.0.weight", "critic.0.weight"):
        old = saved["model_state_dict"][key]
        shape = current[key].shape
        if old.shape == shape:
            continue
        if old.shape[0] != shape[0] or old.shape[1] >= shape[1]:
            raise ValueError("Only appended observations are supported on resume")
        changes[tuple(old.shape)] = tuple(shape)
        value = torch.zeros(shape, dtype=old.dtype)
        value[:, : old.shape[1]] = old
        saved["model_state_dict"][key] = value
    if not changes:
        return Path(checkpoint)
    for state in saved["optimizer_state_dict"]["state"].values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor) and tuple(value.shape) in changes:
                extended = torch.zeros(changes[tuple(value.shape)], dtype=value.dtype)
                extended[:, : value.shape[1]] = value
                state[key] = extended
    width = current["actor.0.weight"].shape[1]
    for name in ("obs_norm_state_dict", "privileged_obs_norm_state_dict"):
        state = saved[name]
        for key, initial in [("_mean", 0.0), ("_var", 0.05**2), ("_std", 0.05)]:
            old = state[key]
            value = torch.full((1, width), initial, dtype=old.dtype)
            value[:, : old.shape[1]] = old
            state[key] = value
    output = Path(output)
    torch.save(saved, output)
    return output
