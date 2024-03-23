import argparse
import os
import sys
import time
from typing import List, Optional

import prettytable as pt
import torch
import yaml
from termcolor import cprint


def load_dataset_arguments(cfg_path, opt):
    if opt.load is None and cfg_path is None:
        return

    # exclude parameters assigned in the command
    if len(sys.argv) > 1:
        arguments = sys.argv[1:]
        arguments = list(
            map(lambda x: x.replace("--", ""), filter(lambda x: "--" in x, arguments))
        )
    else:
        arguments = []

    # load parameters in the yaml file
    if cfg_path is not None:
        opt.load = cfg_path
    else:
        assert os.path.exists(opt.load)
    with open(opt.load, "r") as f:
        yaml_arguments = yaml.safe_load(f)
    # TODO this should be verified
    for k, v in yaml_arguments.items():
        if not k in arguments:
            setattr(opt, k, v)


def get_opt(cfg_path: Optional[str] = None, additional_parsers: Optional[List] = None):
    parents = [get_arguments_parser()]
    if additional_parsers:
        parents.extend(additional_parsers)
    parser = argparse.ArgumentParser(
        "Options for training and evaluation", parents=parents, allow_abbrev=False
    )
    opt = parser.parse_known_args()[0]

    # load dataset argument file
    load_dataset_arguments(cfg_path, opt)

    # user-defined warnings and assertions
    if opt.decoder.lower() not in ["c1"]:
        cprint("Not supported yet! Check if the output use log_softmax!", "red")
        time.sleep(3)

    if opt.map_mask_weight > 0.0 or opt.volume_mask_weight > 0.0:
        cprint("Mask loss is not 0!", "red")
        time.sleep(3)

    if opt.val_set != "val":
        cprint(f"Evaluating on {opt.val_set} set!", "red")
        time.sleep(3)

    if opt.mvc_spixel:
        assert (
            not opt.loss_on_mid_map
        ), "Middle map supervision is not supported with spixel!"

    if "early" in opt.modality:
        assert (
            len(opt.modality) == 1
        ), "Early fusion is not supported for multi-modality!"
    for modal in opt.modality:
        assert modal in [
            "rgb",
            "srm",
            "bayar",
            "early",
        ], f"Unsupported modality {modal}!"

    if opt.resume:
        assert os.path.exists(opt.resume)

    # if opt.mvc_weight <= 0. and opt.consistency_weight > 0.:
    #     assert opt.consistency_source == 'self', 'Ensemble consistency is not supported when mvc_weight is 0!'

    # automatically set parameters
    if len(sys.argv) > 1:
        arguments = sys.argv[1:]
        arguments = list(
            map(lambda x: x.replace("--", ""), filter(lambda x: "--" in x, arguments))
        )
        params = []
        for argument in arguments:
            if not argument in [
                "suffix",
                "save_root_path",
                "dataset",
                "source",
                "resume",
                "num_workers",
                "eval_freq",
                "print_freq",
                "lr_steps",
                "rgb_resume",
                "srm_resume",
                "bayar_resume",
                "teacher_resume",
                "occ",
                "load",
                "amp_opt_level",
                "val_shuffle",
                "tile_size",
                "modality",
            ]:
                try:
                    value = (
                        str(eval("opt.{}".format(argument.split("=")[0])))
                        .replace("[", "")
                        .replace("]", "")
                        .replace(" ", "-")
                        .replace(",", "")
                    )
                    params.append(
                        argument.split("=")[0].replace("_", "").replace(" ", "")
                        + "="
                        + value
                    )
                except:
                    cprint("Unknown argument: {}".format(argument), "red")
            if "early" in opt.modality:
                params.append("modality=early")
        test_name = "_".join(params)

    else:
        test_name = ""

    time_stamp = time.strftime("%b-%d-%H-%M-%S", time.localtime())
    dir_name = "{}_{}{}_{}".format(
        "-".join(list(opt.train_datalist.keys())).upper(),
        test_name,
        opt.suffix,
        time_stamp,
    ).replace("__", "_")

    opt.time_stamp = time_stamp
    opt.dir_name = dir_name
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt.debug or opt.wholetest:
        opt.val_shuffle = True
        cprint("Setting val_shuffle to True in debug and wholetest mode!", "red")
        time.sleep(3)

    if len(opt.modality) < 2 and opt.mvc_weight != 0.0:
        opt.mvc_weight = 0.0
        cprint(
            "Setting multi-view consistency weight to 0. for single modality training",
            "red",
        )
        time.sleep(3)

    if "early" in opt.modality:
        opt.mvc_single_weight = {"early": 1.0}
    else:
        if "rgb" not in opt.modality:
            opt.mvc_single_weight[0] = 0.0
        if "srm" not in opt.modality:
            opt.mvc_single_weight[1] = 0.0
        if "bayar" not in opt.modality:
            opt.mvc_single_weight[2] = 0.0
        weight_sum = sum(opt.mvc_single_weight)
        single_weight = list(map(lambda x: x / weight_sum, opt.mvc_single_weight))
        opt.mvc_single_weight = {
            "rgb": single_weight[0],
            "srm": single_weight[1],
            "bayar": single_weight[2],
        }
    cprint(
        "Change mvc single modality weight to {}".format(opt.mvc_single_weight), "blue"
    )
    time.sleep(3)

    # print parameters
    tb = pt.PrettyTable(field_names=["Arguments", "Values"])
    for k, v in vars(opt).items():
        # some parameters might be too long to display
        if k not in ["dir_name", "resume", "rgb_resume", "srm_resume", "bayar_resume"]:
            tb.add_row([k, v])
    print(tb)

    return opt


def get_arguments_parser():
    parser = argparse.ArgumentParser(
        "CVPR2022 image manipulation detection model", add_help=False
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--wholetest", action="store_true", default=False)

    parser.add_argument(
        "--load", default="configs/final.yaml", help="Load configuration YAML file."
    )
    parser.add_argument("--num_class", type=int, default=1, help="Use sigmoid.")

    # loss-related
    parser.add_argument("--map_label_weight", type=float, default=1.0)
    parser.add_argument("--volume_label_weight", type=float, default=1.0)
    parser.add_argument(
        "--map_mask_weight",
        type=float,
        default=0.0,
        help="Only use this for debug purpose.",
    )
    parser.add_argument(
        "--volume_mask_weight",
        type=float,
        default=0.0,
        help="Only use this for debug purpose.",
    )
    parser.add_argument(
        "--consistency_weight",
        type=float,
        default=0.0,
        help="Consitency between output map and volume within a single view.",
    )
    parser.add_argument(
        "--consistency_type", type=str, default="l2", choices=["l1", "l2"]
    )
    parser.add_argument(
        "--consistency_kmeans",
        action="store_true",
        default=False,
        help="Perform k-means on the volume to determine pristine and modified areas.",
    )
    parser.add_argument(
        "--consistency_stop_map_grad",
        action="store_true",
        default=False,
        help="Stop gradient for the map.",
    )
    parser.add_argument(
        "--consistency_source", type=str, default="self", choices=["self", "ensemble"]
    )
    parser.add_argument("--map_entropy_weight", type=float, default=0.0)
    parser.add_argument("--volume_entropy_weight", type=float, default=0.0)
    parser.add_argument("--mvc_weight", type=float, default=0.0)
    parser.add_argument(
        "--mvc_time_dependent",
        action="store_true",
        default=False,
        help="Use Gaussian smooth on the MVCW weight.",
    )
    parser.add_argument("--mvc_soft", action="store_true", default=False)
    parser.add_argument("--mvc_zeros_on_au", action="store_true", default=False)
    parser.add_argument(
        "--mvc_single_weight",
        type=float,
        nargs="+",
        default=[1.0, 1.0, 1.0],
        help="Weight for the RGB, SRM and Bayar modality for MVC training.",
    )
    parser.add_argument(
        "--mvc_steepness", type=float, default=5.0, help="The large the slower."
    )
    parser.add_argument("--mvc_spixel", action="store_true", default=False)
    parser.add_argument("--mvc_num_spixel", type=int, default=100)
    parser.add_argument(
        "--loss_on_mid_map",
        action="store_true",
        default=False,
        help="This only applies for the output map, but not for the consistency volume.",
    )
    parser.add_argument(
        "--label_loss_on_whole_map",
        action="store_true",
        default=False,
        help="Apply cls loss on the avg(map) for pristine images, instead of max(map).",
    )

    # network architecture
    parser.add_argument("--modality", type=str, default=["rgb"], nargs="+")
    parser.add_argument("--srm_clip", type=float, default=5.0)
    parser.add_argument("--bayar_magnitude", type=float, default=1.0)
    parser.add_argument("--encoder", type=str, default="ResNet50")
    parser.add_argument("--encoder_weight", type=str, default="")
    parser.add_argument("--decoder", type=str, default="C1")
    parser.add_argument("--decoder_weight", type=str, default="")
    parser.add_argument(
        "--fc_dim",
        type=int,
        default=2048,
        help="Changing this might leads to error in the conjunction between encoder and decoder.",
    )
    parser.add_argument(
        "--volume_block_idx",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Compute the consistency volume at certain block.",
    )
    parser.add_argument("--share_embed_head", action="store_true", default=False)
    parser.add_argument(
        "--fcn_up",
        type=int,
        default=32,
        choices=[8, 16, 32],
        help="FCN architecture, 32s, 16s, or 8s.",
    )
    parser.add_argument("--gem", action="store_true", default=False)
    parser.add_argument("--gem_coef", type=float, default=100)
    parser.add_argument("--gsm", action="store_true", default=False)
    parser.add_argument(
        "--map_portion",
        type=float,
        default=0,
        help="Select topk portion of the output map for the image-level classification. 0 for use max.",
    )
    parser.add_argument("--otsu_sel", action="store_true", default=False)
    parser.add_argument("--otsu_portion", type=float, default=1.0)

    # training parameters
    parser.add_argument("--no_gaussian_blur", action="store_true", default=False)
    parser.add_argument("--no_color_jitter", action="store_true", default=False)
    parser.add_argument("--no_jpeg_compression", action="store_true", default=False)
    parser.add_argument("--resize_aug", action="store_true", default=False)
    parser.add_argument(
        "--uncorrect_label",
        action="store_true",
        default=False,
        help="This will not correct image-level labels caused by image cropping.",
    )
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["sgd", "adamw"]
    )
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument(
        "--val_set",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Change to train for debug purpose.",
    )
    parser.add_argument(
        "--val_shuffle", action="store_true", default=False, help="Shuffle val set."
    )
    parser.add_argument("--save_figure", action="store_true", default=False)
    parser.add_argument("--figure_path", type=str, default="figures")
    parser.add_argument("--batch_size", type=int, default=36)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--eval_freq", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=36)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    # lr
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine"',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std-dev (default: 1.0)",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=2e-7,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=2e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )
    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=20,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=5,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10",
    )
    parser.add_argument(
        "--decay-rate",
        "-dr",
        type=float,
        default=0.5,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )
    parser.add_argument("--lr_cycle_limit", "-lcl", type=int, default=1)
    parser.add_argument("--lr_cycle_mul", "-lcm", type=float, default=1)

    # inference hyperparameters
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument(
        "-lis",
        "--large_image_strategy",
        choices=["rescale", "slide", "none"],
        default="slide",
        help="Slide will get better performance than rescale.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=768,
        help="If the testing image is larger than tile_size, I will use sliding window to do the inference.",
    )
    parser.add_argument("--tile_overlap", type=float, default=0.1)
    parser.add_argument("--spixel_postproc", action="store_true", default=False)
    parser.add_argument("--convcrf_postproc", action="store_true", default=False)
    parser.add_argument("--convcrf_shape", type=int, default=512)
    parser.add_argument("--crf_postproc", action="store_true", default=False)
    parser.add_argument("--max_pool_postproc", type=int, default=1)
    parser.add_argument("--crf_downsample", type=int, default=1)
    parser.add_argument("--crf_iter_max", type=int, default=5)
    parser.add_argument("--crf_pos_w", type=int, default=3)
    parser.add_argument("--crf_pos_xy_std", type=int, default=1)
    parser.add_argument("--crf_bi_w", type=int, default=4)
    parser.add_argument("--crf_bi_xy_std", type=int, default=67)
    parser.add_argument("--crf_bi_rgb_std", type=int, default=3)

    # save
    parser.add_argument("--save_root_path", type=str, default="tmp")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--print_freq", type=int, default=100)

    # misc
    parser.add_argument("--seed", type=int, default=1)

    return parser
