"""Short CUDA lifetime/reset test, also suitable for compute-sanitizer."""

import sys, json, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bootstrap
import torch
from duck_gym.tensor_env import TensorEnv

model = Path(os.environ["DUCK_CUDA_BUILD"]).parents[1] / "build/models/standing"
e = TensorEnv(
    model, num_envs=4, iterations=50, randomize=False, episode_seconds=0.04, task="locomotion"
)
initial = e.native.state()[0].clone()
assert e.native.contact_forces().abs().sum() == 0
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    target = e.home.repeat(4, 1)
    target[1, 0] += 0.03
    forces = torch.zeros(4, 3, device=e.device)
    forces[1, 0] = 0.02
    e.native.step(target, forces, 5, 0.55, 0.96)
    del target, forces
forces_snapshot = e.native.contact_forces()
assert torch.isfinite(forces_snapshot).all() and (forces_snapshot[:, :, 2] >= 0).all()
forces_snapshot.fill_(999)
assert not (e.native.contact_forces() == 999).all()
before = e.native.state()[0]
e.reset(torch.tensor([True, False, False, False], device=e.device))
after = e.native.state()[0]
torch.testing.assert_close(after[0], initial[0], rtol=0, atol=1e-7)
torch.testing.assert_close(after[1:], before[1:], rtol=0, atol=0)
for target in [
    e.home.repeat(4, 1).double(),
    e.home.repeat(4, 1)[:, :-1],
    e.home.repeat(4, 1).cpu(),
]:
    try:
        e.native.step(target, e.forces, 1, 0.55, 0.96)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Invalid input accepted")
bad = e.home.repeat(4, 1)
bad[2, 0] = float("nan")
e.native.step(bad, e.forces, 1, 0.55, 0.96)
assert e.native.state()[2][:, 5].tolist() == [0, 0, 1, 0]
e.reset(torch.ones(4, device=e.device, dtype=torch.bool))
e.episode_length_buf[0] = 1
obs, reward, done, info = e.step(torch.zeros(4, 14, device=e.device))
assert done.tolist() == [True, False, False, False]
assert info["time_outs"].tolist() == [True, False, False, False]
assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
torch.cuda.synchronize()
print(json.dumps(dict(boundary_passed=True, gpu=torch.cuda.get_device_name())))
