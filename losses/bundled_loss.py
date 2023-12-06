import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class BundledLoss(nn.Module):
    def __init__(
        self,
        single_modality_loss,
        multi_view_consistency_loss,
        volume_mask_loss,
        multi_view_consistency_weight: float,
        mvc_time_dependent: bool,
        mvc_steepness: float,
        modality: List,
        consistency_weight: float,
        consistency_source: str,
    ):
        super().__init__()

        self.single_modality_loss = single_modality_loss
        self.multi_view_consistency_loss = multi_view_consistency_loss
        self.volume_mask_loss = volume_mask_loss

        self.mvc_weight = multi_view_consistency_weight
        self.mvc_time_dependent = mvc_time_dependent
        self.mvc_steepness = mvc_steepness
        self.modality = modality
        self.consistency_weight = consistency_weight
        self.consistency_source = consistency_source

    def forward(
        self,
        output: Dict,
        label,
        mask,
        epoch: int = 1,
        max_epoch: int = 70,
        spixel=None,
        raw_image=None,
    ):

        total_loss = 0.0
        loss_dict = {}
        for modality in self.modality:
            single_loss = self.single_modality_loss(output[modality], label, mask)

            for k, v in single_loss.items():
                loss_dict[f"{k}/{modality}"] = v
            total_loss = total_loss + single_loss["total_loss"]

        if self.mvc_time_dependent:
            mvc_weight = self.mvc_weight * math.exp(
                -self.mvc_steepness * (1 - epoch / max_epoch) ** 2
            )
        else:
            mvc_weight = self.mvc_weight

        multi_view_consistency_loss = self.multi_view_consistency_loss(
            output, label, spixel, raw_image, mask
        )
        for k, v in multi_view_consistency_loss.items():
            if k not in ["total_loss", "tgt_map"]:
                loss_dict.update({k: v})

        if self.consistency_weight != 0.0 and self.consistency_source == "ensemble":
            for modality in self.modality:
                consisitency_loss = self.volume_mask_loss(
                    output[modality]["out_vol"], multi_view_consistency_loss["tgt_map"]
                )
                consisitency_loss = consisitency_loss["loss"]
                loss_dict[f"consistency_loss/{modality}"] = consisitency_loss
                total_loss = (
                    total_loss
                    + self.consistency_weight
                    * consisitency_loss
                    * math.exp(-self.mvc_steepness * (1 - epoch / max_epoch) ** 2)
                )

        total_loss = total_loss + mvc_weight * multi_view_consistency_loss["total_loss"]

        return {"total_loss": total_loss, **loss_dict}
