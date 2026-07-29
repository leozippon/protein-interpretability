"""ConstrainedAdam optimizer and learning rate scheduler.

ConstrainedAdam maintains unit-norm columns on constrained parameters
(SAE decoder weights) by projecting gradients and re-normalizing after each step.

Reference: InterPLM ConstrainedAdam (external/InterPLM/interplm/train/trainers/common.py)
"""

import math
import torch
from torch.optim import Adam


class ConstrainedAdam(Adam):
    """Adam optimizer with unit-norm constraint on specified parameters.

    Before each step: removes gradient component parallel to parameter directions.
    After each step: re-normalizes constrained parameters to unit norm.
    """

    def __init__(self, params, constrained_params: list[torch.nn.Parameter], **kwargs):
        super().__init__(params, **kwargs)
        self._constrained = constrained_params

    @torch.no_grad()
    def step(self, closure=None):
        # Project gradients for constrained params
        for p in self._constrained:
            if p.grad is None:
                continue
            W = p.data
            G = p.grad
            norms = W.norm(dim=1, keepdim=True).clamp(min=1e-8)
            W_normed = W / norms
            parallel = (G * W_normed).sum(dim=1, keepdim=True)
            G -= parallel * W_normed

        loss = super().step(closure)

        # Re-normalize constrained params
        for p in self._constrained:
            norms = p.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
            p.data /= norms

        return loss


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    decay_start: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing with linear warmup.

    Schedule:
      [0, warmup_steps):       linear warmup 0 → 1
      [warmup_steps, decay_start): constant 1.0
      [decay_start, total_steps]:  cosine decay 1.0 → min_lr_ratio
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / max(total_steps - decay_start, 1)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
