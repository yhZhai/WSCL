import torch
import torch.nn as nn
from einops import rearrange


def get_volume_mask_loss(opt):
    return VolumeMaskLoss()


class VolumeMaskLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce_loss = nn.BCELoss(reduction="mean")

    def _get_volume_mask(self, mask):
        with torch.no_grad():
            h, w = mask.shape[-2:]
            # use orthogonal vector [0, 1] and [1, 0] to generate the ground truth
            mask[torch.where(mask > 0.5)] = 1.0
            mask[torch.where(mask <= 0.5)] = 0.0

            mask = rearrange(mask, "b c h w -> b c (h w)")
            mask_append = 1 - mask.clone()
            mask = torch.cat([mask, mask_append], dim=1)
            mask = torch.bmm(mask.transpose(-1, -2), mask)
            mask = rearrange(mask, "b (h1 w1) (h2 w2) -> b h1 w1 h2 w2", h1=h, h2=h)
            mask = 1 - mask  # 0 indicates consistency, and 1 indicates inconsistency
        return mask

    def forward(self, out_volume, mask):
        volume_size = out_volume.shape[-2:]
        if volume_size != mask.shape[-2:]:
            mask = nn.functional.interpolate(
                mask, size=volume_size, mode="bilinear", align_corners=False
            )
        volume_mask = self._get_volume_mask(mask)
        loss = self.bce_loss(out_volume, volume_mask)

        return {"loss": loss}
