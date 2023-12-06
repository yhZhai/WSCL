from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


class MainModel(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        fc_dim: int,
        volume_block_idx: int,
        share_embed_head: bool,
        pre_filter=None,
        use_gem: bool = False,
        gem_coef: Optional[float] = None,
        use_gsm: bool = False,
        map_portion: float = 0,
        otsu_sel: bool = False,
        otsu_portion: float = 1,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_gem = use_gem
        self.gem_coef = gem_coef
        self.use_gsm = use_gsm
        self.map_portion = map_portion
        assert self.map_portion <= 0.5, "Map_portion must be less than 0.5"
        self.otsu_sel = otsu_sel
        self.otsu_portion = otsu_portion

        self.volume_block_idx = volume_block_idx
        volume_in_channel = int(fc_dim * (2 ** (self.volume_block_idx - 3)))
        volume_out_channel = volume_in_channel // 2

        self.scale = volume_out_channel**0.5
        self.share_embed_head = share_embed_head
        self.proj_head1 = nn.Sequential(
            nn.Conv2d(
                volume_in_channel, volume_in_channel, kernel_size=1, stride=1, padding=0
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                volume_in_channel,
                volume_out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        if not share_embed_head:
            self.proj_head2 = nn.Sequential(
                nn.Conv2d(
                    volume_in_channel,
                    volume_in_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.LeakyReLU(),
                nn.Conv2d(
                    volume_in_channel,
                    volume_out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )

        self.pre_filter = pre_filter

    def forward(self, image, seg_size=None):
        """
        for output maps, the return value is the raw logits
        for consistency volume, the return value is the value after sigmoid
        """
        bs = image.shape[0]
        if self.pre_filter is not None:
            image = self.pre_filter(image)

        # get output map
        encoder_feature = self.encoder(image, return_feature_maps=True)
        output_map = self.decoder(encoder_feature, segSize=seg_size)
        output_map = output_map.sigmoid()
        # b, _, h, w = output_map.shape

        # get image-level prediction
        if self.use_gem:
            mh, mw = output_map.shape[-2:]
            image_pred = output_map.flatten(1)
            image_pred = torch.linalg.norm(image_pred, ord=self.gem_coef, dim=1)
            image_pred = image_pred / (mh * mw)
        elif self.use_gsm:
            image_pred = output_map.flatten(1)
            weight = project_onto_l1_ball(image_pred, 1.0)
            image_pred = (image_pred * weight).sum(1)
        else:
            if self.otsu_sel:
                n_pixel = output_map.shape[-1] * output_map.shape[-2]
                image_pred = output_map.flatten(1)
                image_pred, _ = torch.sort(image_pred, dim=1)
                tmp = []
                for b in range(bs):
                    num_otsu_sel = get_otsu_k(image_pred[b, ...], sorted=True)
                    num_otsu_sel = max(num_otsu_sel, n_pixel // 2 + 1)
                    tpk = int(max(1, (n_pixel - num_otsu_sel) * self.otsu_portion))
                    topk_output = torch.topk(image_pred[b, ...], k=tpk, dim=0)[0]
                    tmp.append(topk_output.mean())
                image_pred = torch.stack(tmp)
            else:
                if self.map_portion == 0:
                    image_pred = nn.functional.max_pool2d(
                        output_map, kernel_size=output_map.shape[-2:]
                    )
                    image_pred = image_pred.squeeze(1).squeeze(1).squeeze(1)
                else:
                    n_pixel = output_map.shape[-1] * output_map.shape[-2]
                    k = int(max(1, int(self.map_portion * n_pixel)))
                    topk_output = torch.topk(output_map.flatten(1), k, dim=1)[0]
                    image_pred = topk_output.mean(1)

        if seg_size is not None:
            output_map = nn.functional.interpolate(
                output_map, size=seg_size, mode="bilinear", align_corners=False
            )
            output_map = output_map.clamp(0, 1)

        # compute consistency volume, 0 for consistency, and 1 for inconsistency
        feature_map1 = self.proj_head1(encoder_feature[self.volume_block_idx])
        if not self.share_embed_head:
            feature_map2 = self.proj_head2(encoder_feature[self.volume_block_idx])
        else:
            feature_map2 = feature_map1.clone()
        b, c, h, w = feature_map1.shape
        feature_map1 = rearrange(feature_map1, "b c h w -> b c (h w)")
        feature_map2 = rearrange(feature_map2, "b c h w -> b c (h w)")
        consistency_volume = torch.bmm(feature_map1.transpose(-1, -2), feature_map2)
        consistency_volume = rearrange(
            consistency_volume, "b (h1 w1) (h2 w2) -> b h1 w1 h2 w2", h1=h, h2=h
        )
        consistency_volume = consistency_volume / self.scale
        consistency_volume = 1 - consistency_volume.sigmoid()

        vh, vw = consistency_volume.shape[-2:]
        if self.use_gem:
            volume_image_pred = consistency_volume.flatten(1)
            volume_image_pred = torch.linalg.norm(
                volume_image_pred, ord=self.gem_coef, dim=1
            )
            volume_image_pred = volume_image_pred / (vh * vw * vh * vw)
        elif self.use_gsm:
            volume_image_pred = consistency_volume.flatten(1)
            weight = project_onto_l1_ball(volume_image_pred, 1.0)
            volume_image_pred = (volume_image_pred * weight).sum(1)
        else:
            # FIXME skip Otsu's selection on volume due to its slowness
            # if self.otsu_sel:
            #     n_ele = vh * vw * vh * vw
            #     volume_image_pred = consistency_volume.flatten(1)
            #     volume_image_pred, _ = torch.sort(volume_image_pred, dim=1)
            #     tmp = []
            #     for b in range(bs):
            #         num_otsu_sel = get_otsu_k(volume_image_pred[b, ...], sorted=True)
            #         num_otsu_sel = max(num_otsu_sel, n_ele // 2 + 1)
            #         tpk = int(max(1, (n_ele - num_otsu_sel) * self.otsu_portion))
            #         topk_output = torch.topk(volume_image_pred[b, ...], k=tpk, dim=0)[0]
            #         tmp.append(topk_output.mean())
            #     volume_image_pred = torch.stack(tmp)
            # else:
            if self.map_portion == 0:
                volume_image_pred = torch.max(consistency_volume.flatten(1), dim=1)[0]
            else:
                n_ele = vh * vw * vh * vw
                k = int(max(1, int(self.map_portion * n_ele)))
                topk_output = torch.topk(consistency_volume.flatten(1), k, dim=1)[0]
                volume_image_pred = topk_output.mean(1)

        return {
            "out_map": output_map,
            "map_pred": image_pred,
            "out_vol": consistency_volume,
            "vol_pred": volume_image_pred,
        }


def project_onto_l1_ball(x, eps):
    """
    Compute Euclidean projection onto the L1 ball for a batch.

      min ||x - u||_2 s.t. ||u||_1 <= eps

    Inspired by the corresponding numpy version by Adrien Gaidon.

    Parameters
    ----------
    x: (batch_size, *) torch array
      batch of arbitrary-size tensors to project, possibly on GPU

    eps: float
      radius of l-1 ball to project onto

    Returns
    -------
    u: (batch_size, *) torch array
      batch of projected tensors, reshaped to match the original

    Notes
    -----
    The complexity of this algorithm is in O(dlogd) as it involves sorting x.

    References
    ----------
    [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
    """
    with torch.no_grad():
        original_shape = x.shape
        x = x.view(x.shape[0], -1)
        mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
        mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
        cumsum = torch.cumsum(mu, dim=1)
        arange = torch.arange(1, x.shape[1] + 1, device=x.device)
        rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
        theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
        proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
        x = mask * x + (1 - mask) * proj * torch.sign(x)
        x = x.view(original_shape)
    return x


def get_otsu_k(attention, return_value=False, sorted=False):
    def _get_weighted_var(seq, pivot: int):
        # seq is of shape [t], in ascending order
        length = seq.shape[0]
        wb = pivot / length
        vb = seq[:pivot].var()
        wf = 1 - pivot / length
        vf = seq[pivot:].var()
        return wb * vb + wf * vf

    # attention shape: t
    # TODO use half
    length = attention.shape[0]
    if length == 1:
        return 0
    elif length == 2:
        return 1
    if not sorted:
        attention, _ = torch.sort(attention)
    optimal_i = length // 2
    min_intra_class_var = _get_weighted_var(attention, optimal_i)

    # for i in range(1, length):
    #     intra_class_var = _get_weighted_var(attention, i)
    #     if intra_class_var < min_intra_class_var:
    #         min_intra_class_var = intra_class_var
    #         optimal_i = i

    got_it = False
    # look left
    for i in range(optimal_i - 1, 0, -1):
        intra_class_var = _get_weighted_var(attention, i)
        if intra_class_var > min_intra_class_var:
            break
        else:
            min_intra_class_var = intra_class_var
            optimal_i = i
            got_it = True
    # look right
    if not got_it:
        for i in range(optimal_i + 1, length):
            intra_class_var = _get_weighted_var(attention, i)
            if intra_class_var > min_intra_class_var:
                break
            else:
                min_intra_class_var = intra_class_var
                optimal_i = i

    if return_value:
        return attention[optimal_i]
    else:
        return optimal_i


if __name__ == "__main__":
    model = MainModel(None, None, 1024, 2, True, "srm")
