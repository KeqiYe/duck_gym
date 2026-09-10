"""Observation extension preserves policy outputs and usable Adam state."""

import sys, tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import torch
from rsl_rl.modules import ActorCritic
from checkpoint_io import extend_observations

torch.manual_seed(42)
a = ActorCritic(58, 58, 14, actor_hidden_dims=[8], critic_hidden_dims=[8])
b = ActorCritic(60, 60, 14, actor_hidden_dims=[8], critic_hidden_dims=[8])
optimizer = torch.optim.Adam(a.parameters())
x = torch.randn(8, 58)
(a.actor(x).square().mean() + a.critic(x).square().mean()).backward()
optimizer.step()
norm = {
    "_mean": torch.zeros(1, 58),
    "_var": torch.ones(1, 58),
    "_std": torch.ones(1, 58),
    "count": torch.tensor(100),
}
with tempfile.TemporaryDirectory() as folder:
    source = Path(folder) / "old.pt"
    out = Path(folder) / "adapted.pt"
    torch.save(
        dict(
            model_state_dict=a.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            obs_norm_state_dict=norm,
            privileged_obs_norm_state_dict=norm,
            iter=0,
        ),
        source,
    )
    extend_observations(source, SimpleNamespace(alg=SimpleNamespace(policy=b)), out)
    saved = torch.load(out, weights_only=False)
    b.load_state_dict(saved["model_state_dict"])
    y = torch.cat((x, torch.randn(8, 2)), dim=-1)
    torch.testing.assert_close(a.actor(x), b.actor(y), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(a.critic(x), b.critic(y), rtol=1e-6, atol=1e-7)
    new_optimizer = torch.optim.Adam(b.parameters())
    new_optimizer.load_state_dict(saved["optimizer_state_dict"])
    (b.actor(y).square().mean() + b.critic(y).square().mean()).backward()
    new_optimizer.step()
    assert saved["obs_norm_state_dict"]["_mean"].shape == (1, 60)
    assert int(saved["obs_norm_state_dict"]["count"]) == 100
print("Checkpoint observation extension and optimizer continuation passed")
