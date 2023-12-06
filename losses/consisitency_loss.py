import torch
import torch.nn as nn
from einops import rearrange
from fast_pytorch_kmeans import KMeans


def get_consistency_loss(opt):
    loss = ConsistencyLoss(
        opt.consistency_type, opt.consistency_kmeans, opt.consistency_stop_map_grad
    )
    return loss


class ConsistencyLoss(nn.Module):
    def __init__(
        self, loss: str, do_kmeans: bool = True, consistency_stop_map_grad: bool = False
    ):
        super().__init__()
        assert loss in ["l1", "l2"]

        if loss == "l1":
            self.consistency_loss = nn.L1Loss(reduction="mean")
        else:  # l2
            self.consistency_loss = nn.MSELoss(reduction="mean")

        self.do_kmeans = do_kmeans
        if do_kmeans:
            self.kmeans = KMeans(2)
        else:
            self.kmeans = None

        self.consistency_stop_map_grad = consistency_stop_map_grad

    def forward(self, out_volume, out_map, label):
        map_shape = out_map.shape[-2:]
        out_volume = get_volume_seg_map(out_volume, map_shape, label, self.kmeans)
        if self.consistency_stop_map_grad:
            loss = self.consistency_loss(out_volume, out_map.detach())
        else:
            loss = self.consistency_loss(out_volume, out_map)
        return {"loss": loss, "out_vol": out_volume.squeeze(1)}


def get_volume_seg_map(volume, size, label, kmeans=None):
    """volume is of shape [b, h, w, h, w], and size is [h', w']"""
    batch_size = volume.shape[0]
    volume_shape = volume.shape[-2:]
    volume = rearrange(volume, "b h1 w1 h2 w2 -> b (h1 w1) (h2 w2)")
    if kmeans is not None:  # do k-means on out_volume
        for i in range(batch_size):
            # NOTE K-means only applies for manipulated images!
            if label[i] == 0:
                continue
            batch_volume = volume[i, ...]
            out = kmeans.fit_predict(batch_volume)
            ones = torch.where(out == 1)
            zeros = torch.where(out == 0)
            if (
                ones[0].numel() >= zeros[0].numel()
            ):  # intuitively, the cluster with fewer elements is the modified cluster
                pristine, modified = ones, zeros
            else:
                pristine, modified = zeros, ones
            volume[i, :, modified[0]] = 1 - volume[i, :, modified[0]]

    volume = volume.mean(dim=-1)
    volume = rearrange(volume, "b (h w) -> b h w", h=volume_shape[0])
    volume = volume.unsqueeze(1)
    if volume_shape != size:
        volume = nn.functional.interpolate(
            volume, size=size, mode="bilinear", align_corners=False
        )
    return volume  # size [b, 1, h, w]
