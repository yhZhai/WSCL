from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from skimage import segmentation


def get_multi_view_consistency_loss(opt):
    loss = MultiViewConsistencyLoss(
        opt.mvc_soft,
        opt.mvc_zeros_on_au,
        opt.mvc_single_weight,
        opt.modality,
        opt.mvc_spixel,
        opt.mvc_num_spixel,
    )
    return loss


class MultiViewConsistencyLoss(nn.Module):
    def __init__(
        self,
        soft: bool,
        zeros_on_au: bool,
        single_weight: Dict,
        modality: List,
        spixel: bool = False,
        num_spixel: int = 100,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.soft = soft
        self.zeros_on_au = zeros_on_au
        self.single_weight = single_weight
        self.modality = modality
        self.spixel = spixel
        self.num_spixel = num_spixel
        self.eps = eps

        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(self, output: Dict, label, spixel=None, image=None, mask=None):

        tgt_map = torch.zeros_like(
            output[self.modality[0]]["out_map"], requires_grad=False
        )
        with torch.no_grad():
            for modality in self.modality:
                weight = self.single_weight[modality.lower()]
                tgt_map = tgt_map + weight * output[modality]["out_map"]

        if self.spixel:
            # raw_tgt_map = tgt_map.clone()
            tgt_map = get_spixel_tgt_map(tgt_map, spixel)

        if not self.soft:
            for b in range(tgt_map.shape[0]):
                if tgt_map[b, ...].max() <= 0.5 and label[b] == 1.0:
                    tgt_map[b, ...][
                        torch.where(tgt_map[b, ...] == torch.max(tgt_map[b, ...]))
                    ] = 1.0
            tgt_map[torch.where(tgt_map > 0.5)] = 1
            tgt_map[torch.where(tgt_map <= 0.5)] = 0
            tgt_map[torch.where(label == 0.0)[0], ...] = 0.0

        if self.zeros_on_au:
            tgt_map[torch.where(label == 0.0)[0], ...] = 0.0

        total_loss = 0.0
        loss_dict = {}
        for modality in self.modality:
            loss = self.mse_loss(output[modality]["out_map"], tgt_map)
            loss_dict[f"multi_view_consistency_loss_{modality}"] = loss
            total_loss = total_loss + loss

        return {**loss_dict, "tgt_map": tgt_map, "total_loss": total_loss}

    def _save(
        self,
        spixel: torch.Tensor,
        image: torch.Tensor,
        mask: torch.Tensor,
        tgt_map: torch.Tensor,
        raw_tgt_map: torch.Tensor,
        out_path: str = "tmp/spixel_tgt_map.png",
    ):
        spixel = spixel.permute(0, 2, 3, 1).detach().cpu().numpy()
        image = image.permute(0, 2, 3, 1).detach().cpu().numpy()
        mask = mask.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0
        tgt_map = tgt_map.permute(0, 2, 3, 1).squeeze(3).detach().cpu().numpy() * 255.0
        raw_tgt_map = (
            raw_tgt_map.permute(0, 2, 3, 1).squeeze(3).detach().cpu().numpy() * 255.0
        )
        bn = spixel.shape[0]
        i = 1
        for b in range(bn):
            plt.subplot(bn, 5, i)
            i += 1
            plt.imshow(image[b])
            plt.axis("off")
            plt.title("image")
            plt.subplot(bn, 5, i)
            i += 1
            plt.imshow(mask[b])
            plt.axis("off")
            plt.title("mask")
            plt.subplot(bn, 5, i)
            i += 1
            plt.imshow(spixel[b])
            plt.axis("off")
            plt.title("superpixel")
            plt.subplot(bn, 5, i)
            i += 1
            plt.imshow(raw_tgt_map[b])
            plt.axis("off")
            plt.title("raw target map")
            plt.subplot(bn, 5, i)
            i += 1
            plt.imshow(tgt_map[b])
            plt.axis("off")
            plt.title("target map")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()


def get_spixel_tgt_map(weighted_sum, spixel):
    b, _, h, w = weighted_sum.shape
    spixel_tgt_map = torch.zeros_like(weighted_sum, requires_grad=False)

    for bidx in range(b):
        spixel_indices = spixel[bidx, ...].unique()
        # num_spixel = spixel_idx.shape[0]
        for spixel_idx in spixel_indices.tolist():
            area = (spixel[bidx, ...] == spixel_idx).sum()
            weighted_sum_in_area = weighted_sum[bidx, ...][
                torch.where(spixel[bidx, ...] == spixel_idx)
            ].sum()
            avg_area = weighted_sum_in_area / area
            # this is soft map, and the threshold process will be conducted in the forward function
            spixel_tgt_map[bidx][
                torch.where(spixel[bidx, ...] == spixel_idx)
            ] = avg_area

    return spixel_tgt_map


if __name__ == "__main__":
    mvc_loss = MultiViewConsistencyLoss(True, True, [1, 1, 2])
    print("a")
