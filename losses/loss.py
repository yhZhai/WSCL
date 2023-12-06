import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(
        self,
        map_label_loss,
        volume_label_loss,
        map_mask_loss,
        volume_mask_loss,
        consistency_loss,
        entropy_loss,
        map_label_weight,
        volume_label_weight,
        map_mask_weight,
        volume_mask_weight,
        consistency_weight,
        map_entropy_weight,
        volume_entropy_weight,
        consistency_source,
    ):
        super().__init__()

        self.map_label_loss = map_label_loss
        self.volume_label_loss = volume_label_loss
        self.map_mask_loss = map_mask_loss
        self.volume_mask_loss = volume_mask_loss
        self.consistency_loss = consistency_loss
        self.entropy_loss = entropy_loss

        self.map_label_weight = map_label_weight
        self.volume_label_weight = volume_label_weight
        self.map_mask_weight = map_mask_weight
        self.volume_mask_weight = volume_mask_weight
        self.consistency_weight = consistency_weight
        self.map_entropy_weight = map_entropy_weight
        self.volume_entropy_weight = volume_entropy_weight
        self.consistency_source = consistency_source

    def forward(self, output, label, mask):
        total_loss = 0.0
        loss_dict = {}

        # --- label loss ---
        label = label.float()
        # compute map label loss anyway
        map_label_loss = self.map_label_loss(
            output["map_pred"], output["out_map"], label
        )["loss"]
        total_loss = total_loss + self.map_label_weight * map_label_loss
        loss_dict.update({"map_label_loss": map_label_loss})

        if self.volume_label_weight != 0.0:
            volume_label_loss = self.volume_label_loss(
                output["vol_pred"], output["out_vol"], label
            )["loss"]
            total_loss = total_loss + self.volume_label_weight * volume_label_loss
            loss_dict.update({"vol_label_loss": volume_label_loss})

        # --- mask loss ---
        # compute map mask loss anyway
        map_mask_loss = self.map_mask_loss(output["out_map"], mask)["loss"]
        total_loss = total_loss + self.map_mask_weight * map_mask_loss
        loss_dict.update({"map_mask_loss": map_mask_loss})

        if self.volume_mask_weight != 0.0:
            volume_mask_loss = self.volume_mask_loss(output["out_vol"], mask)["loss"]
            total_loss = total_loss + self.volume_mask_weight * volume_mask_loss
            loss_dict.update({"vol_mask_loss": volume_mask_loss})

        # --- self-consistency loss ---
        if self.consistency_weight != 0.0 and self.consistency_source == "self":
            consistency_loss = self.consistency_loss(
                output["out_vol"], output["out_map"], label
            )
            consistency_loss = consistency_loss["loss"]
            total_loss = total_loss + self.consistency_weight * consistency_loss
            loss_dict.update({"consistency_loss": consistency_loss})

        # --- entropy loss ---
        if self.map_entropy_weight != 0.0:
            map_entropy_loss = self.entropy_loss(output["out_map"])["loss"]
            total_loss = total_loss + self.map_entropy_weight * map_entropy_loss
            loss_dict.update({"map_entropy_loss": map_entropy_loss})

        if self.volume_entropy_weight != 0:
            volume_entropy_loss = self.entropy_loss(output["out_vol"])["loss"]
            total_loss = total_loss + self.volume_entropy_weight * volume_entropy_loss
            loss_dict.update({"vol_entropy_loss": volume_entropy_loss})

        loss_dict.update({"total_loss": total_loss})
        return loss_dict
