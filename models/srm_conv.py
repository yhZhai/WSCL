import numpy as np
import torch
import torch.nn as nn


class SRMConv2d(nn.Module):
    def __init__(self, stride: int = 1, padding: int = 2, clip: float = 2):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.clip = clip
        self.conv = self._get_srm_filter()

    def _get_srm_filter(self):
        filter1 = [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ]
        filter2 = [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ]
        filter3 = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        q = [4.0, 12.0, 2.0]
        filter1 = np.asarray(filter1, dtype=float) / q[0]
        filter2 = np.asarray(filter2, dtype=float) / q[1]
        filter3 = np.asarray(filter3, dtype=float) / q[2]
        filters = [
            [filter1, filter1, filter1],
            [filter2, filter2, filter2],
            [filter3, filter3, filter3],
        ]
        filters = torch.tensor(filters).float()
        conv2d = nn.Conv2d(
            3,
            3,
            kernel_size=5,
            stride=self.stride,
            padding=self.padding,
            padding_mode="zeros",
        )
        conv2d.weight = nn.Parameter(filters, requires_grad=False)
        conv2d.bias = nn.Parameter(torch.zeros_like(conv2d.bias), requires_grad=False)
        return conv2d

    def forward(self, x):
        x = self.conv(x)
        if self.clip != 0.0:
            x = x.clamp(-self.clip, self.clip)
        return x


if __name__ == "__main__":
    srm = SRMConv2d()
    x = torch.rand((63, 3, 64, 64))
    x = srm(x)
