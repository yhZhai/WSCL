import torch
import torch.nn as nn
from einops import rearrange


class BayarConv2d(nn.Module):
    def __init__(
        self,
        in_channles: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 0,
        magnitude: float = 1.0,
    ):
        super().__init__()
        assert kernel_size > 1, "Bayar conv kernel size must be greater than 1"

        self.in_channels = in_channles
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.magnitude = magnitude

        self.center_weight = nn.Parameter(
            torch.ones(self.in_channels, self.out_channels, 1) * -1.0 * magnitude,
            requires_grad=False,
        )
        self.kernel_weight = nn.Parameter(
            torch.rand((self.in_channels, self.out_channels, kernel_size**2 - 1)),
            requires_grad=True,
        )

    def _constraint_weight(self):
        self.kernel_weight.data = self.kernel_weight.permute(2, 0, 1)
        self.kernel_weight.data = torch.div(
            self.kernel_weight.data, self.kernel_weight.data.sum(0)
        )
        self.kernel_weight.data = self.kernel_weight.permute(1, 2, 0) * self.magnitude
        center_idx = self.kernel_size**2 // 2
        full_kernel = torch.cat(
            [
                self.kernel_weight[:, :, :center_idx],
                self.center_weight,
                self.kernel_weight[:, :, center_idx:],
            ],
            dim=2,
        )
        full_kernel = rearrange(
            full_kernel, "ci co (kw kh) -> ci co kw kh", kw=self.kernel_size
        )
        return full_kernel

    def forward(self, x):
        x = nn.functional.conv2d(
            x, self._constraint_weight(), stride=self.stride, padding=self.padding
        )
        return x


if __name__ == "__main__":
    device = "cuda"
    bayer_conv2d = BayarConv2d(3, 3, 3, magnitude=1).to(device)
    bayer_conv2d._constraint_weight()
    i = torch.rand(16, 3, 16, 16).to(device)
    o = bayer_conv2d(i)
