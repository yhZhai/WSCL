import torch
import torch.nn as nn


def get_entropy_loss(opt):
    return EntropyLoss()


class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.exp = 1e-7
        assert self.exp < 0.5

    def forward(self, item):
        item = item.clamp(min=self.exp, max=1 - self.exp)
        entropy = -item * torch.log(item) - (1 - item) * torch.log(1 - item)
        entropy = entropy.mean()

        return {"loss": entropy}
