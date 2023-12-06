import torch.nn as nn

from .bayar_conv import BayarConv2d
from .early_fusion_pre_filter import EarlyFusionPreFilter
from .ensemble_model import EnsembleModel
from .main_model import MainModel
from .models import ModelBuilder, SegmentationModule
from .srm_conv import SRMConv2d


def get_ensemble_model(opt):
    models = {}
    for modality in opt.modality:
        models[modality] = get_single_modal_model(opt, modality)

    ensemble_model = EnsembleModel(
        models=models, mvc_single_weight=opt.mvc_single_weight
    )
    return ensemble_model


def get_single_modal_model(opt, modality):
    encoder = ModelBuilder.build_encoder(  # TODO check the implementation of FCN
        arch=opt.encoder.lower(), fc_dim=opt.fc_dim, weights=opt.encoder_weight
    )
    decoder = ModelBuilder.build_decoder(
        arch=opt.decoder.lower(),
        fc_dim=opt.fc_dim,
        weights=opt.decoder_weight,
        num_class=opt.num_class,
        dropout=opt.dropout,
        fcn_up=opt.fcn_up,
    )

    if modality.lower() == "bayar":
        pre_filter = BayarConv2d(
            3, 3, 5, stride=1, padding=2, magnitude=opt.bayar_magnitude
        )
    elif modality.lower() == "srm":
        pre_filter = SRMConv2d(
            stride=1, padding=2, clip=opt.srm_clip
        )  # TODO check the implementation of SRM filter
    elif modality.lower() == "rgb":
        pre_filter = nn.Identity()
    else:  # early
        pre_filter = EarlyFusionPreFilter(
            bayar_magnitude=opt.bayar_magnitude, srm_clip=opt.srm_clip
        )

    model = MainModel(
        encoder,
        decoder,
        opt.fc_dim,
        opt.volume_block_idx,
        opt.share_embed_head,
        pre_filter,
        opt.gem,
        opt.gem_coef,
        opt.gsm,
        opt.map_portion,
        opt.otsu_sel,
        opt.otsu_portion,
    )

    return model
