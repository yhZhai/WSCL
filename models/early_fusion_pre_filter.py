import torch
import torch.nn as nn

from .bayar_conv import BayarConv2d
from .srm_conv import SRMConv2d


class EarlyFusionPreFilter(nn.Module):
    def __init__(self, bayar_magnitude: float, srm_clip: float):
        super().__init__()
        self.bayar_filter = BayarConv2d(
            3, 3, 5, stride=1, padding=2, magnitude=bayar_magnitude
        )
        self.srm_filter = SRMConv2d(stride=1, padding=2, clip=srm_clip)
        self.rgb_filter = nn.Identity()
        self.map = nn.Conv2d(9, 3, 1, stride=1, padding=0)

    def forward(self, x):
        x_bayar = self.bayar_filter(x)
        x_srm = self.srm_filter(x)
        x_rgb = self.rgb_filter(x)

        x_concat = torch.cat([x_bayar, x_srm, x_rgb], dim=1)
        x_concat = self.map(x_concat)
        return x_concat
