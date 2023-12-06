from .bundled_loss import BundledLoss
from .consisitency_loss import get_consistency_loss, get_volume_seg_map
from .entropy_loss import get_entropy_loss
from .loss import Loss
from .map_label_loss import get_map_label_loss
from .map_mask_loss import get_map_mask_loss
from .multi_view_consistency_loss import (
    get_multi_view_consistency_loss,
    get_spixel_tgt_map,
)
from .volume_label_loss import get_volume_label_loss
from .volume_mask_loss import get_volume_mask_loss


def get_bundled_loss(opt):
    """Loss function for the overeall training, including the multi-view
    consistency loss."""
    single_modality_loss = get_loss(opt)
    multi_view_consistency_loss = get_multi_view_consistency_loss(opt)
    volume_mask_loss = get_volume_mask_loss(opt)
    bundled_loss = BundledLoss(
        single_modality_loss,
        multi_view_consistency_loss,
        volume_mask_loss,
        opt.mvc_weight,
        opt.mvc_time_dependent,
        opt.mvc_steepness,
        opt.modality,
        opt.consistency_weight,
        opt.consistency_source,
    )

    return bundled_loss


def get_loss(opt):
    """Loss function for a single model, excluding the multi-view consistency
    loss."""
    map_label_loss = get_map_label_loss(opt)
    volume_label_loss = get_volume_label_loss(opt)
    map_mask_loss = get_map_mask_loss(opt)
    volume_mask_loss = get_volume_mask_loss(opt)
    consisitency_loss = get_consistency_loss(opt)
    entropy_loss = get_entropy_loss(opt)
    loss = Loss(
        map_label_loss,
        volume_label_loss,
        map_mask_loss,
        volume_mask_loss,
        consisitency_loss,
        entropy_loss,
        opt.map_label_weight,
        opt.volume_label_weight,
        opt.map_mask_weight,
        opt.volume_mask_weight,
        opt.consistency_weight,
        opt.map_entropy_weight,
        opt.volume_entropy_weight,
        opt.consistency_source,
    )
    return loss
