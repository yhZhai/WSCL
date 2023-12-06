import torch
import torch.nn as nn


def get_map_mask_loss(opt):
    return MapMaskLoss()


class MapMaskLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce_loss = nn.BCELoss(reduction="mean")

    def forward(self, out_map, mask):
        mask_size = mask.shape[-2:]
        if out_map.shape[-2:] != mask_size:
            out_map = nn.functional.interpolate(
                out_map, size=mask_size, mode="bilinear", align_corners=False
            )
        loss = self.bce_loss(out_map, mask)
        return {"loss": loss}


if __name__ == "__main__":
    map_mask_loss = MapMaskLoss()
