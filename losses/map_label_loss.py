import torch
import torch.nn as nn


def get_map_label_loss(opt):
    return MapLabelLoss(opt.label_loss_on_whole_map)


class MapLabelLoss(nn.Module):
    def __init__(self, label_loss_on_whole_map=False):
        super().__init__()

        self.bce_loss = nn.BCELoss(reduction="none")
        self.label_loss_on_whole_map = label_loss_on_whole_map

    def forward(self, pred, out_map, label):
        batch_size = label.shape[0]
        if (
            self.label_loss_on_whole_map
        ):  # apply the loss on the whole map for pristine images
            total_loss = 0
            for i in range(batch_size):
                if label[i] == 0:  # pristine
                    total_loss = (
                        total_loss
                        + self.bce_loss(out_map[i, ...].mean(), label[i]).mean()
                    )
                else:  # modified
                    total_loss = total_loss + self.bce_loss(pred[i], label[i]).mean()
            loss = total_loss / batch_size
        else:
            loss = self.bce_loss(pred, label)
            loss = loss.mean()
        return {"loss": loss}
