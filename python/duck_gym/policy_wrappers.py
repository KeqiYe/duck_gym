"""Deployment wrappers that preserve a training task's action semantics."""

import torch


class NeutralHeadPolicy(torch.nn.Module):
    def __init__(self, policy, head_ids):
        super().__init__()
        self.policy = policy
        mask = torch.ones(14)
        mask[head_ids] = 0
        self.register_buffer("action_mask", mask)

    def forward(self, observation):
        return self.policy(observation) * self.action_mask
