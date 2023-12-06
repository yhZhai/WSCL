import torch.nn as nn


def get_volume_label_loss(opt):
    return VolumeLabelLoss()


class VolumeLabelLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.BCE_loss = nn.BCELoss(reduction="mean")

    def forward(self, pred, volume, label):
        loss = self.BCE_loss(pred, label)
        return {"loss": loss}
