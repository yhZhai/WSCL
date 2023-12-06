import itertools
import os
import random
import shutil
from math import ceil
from typing import Dict, List

import numpy as np
import prettytable as pt
import torch
import torch.nn as nn
from fast_pytorch_kmeans import KMeans
from pathlib2 import Path
from scipy.stats import hmean
from sklearn import metrics
from termcolor import cprint
from torchvision.utils import draw_segmentation_masks, make_grid, save_image

import utils.misc as misc
from losses import get_spixel_tgt_map, get_volume_seg_map
from utils.convcrf import convcrf
from utils.crf import DenseCRF


def train(
    model: nn.Module,
    dataloader,
    dataset_title: str,
    optimizer_dict: Dict,
    criterion,
    epoch: int,
    writer,
    suffix: str,
    opt,
):

    metric_logger = misc.MetricLogger(writer=writer, suffix=suffix)
    cprint("{}-th epoch training on {}".format(epoch, dataset_title), "blue")
    model.train()
    roc_auc_elements = {
        modality: {"map_scores": [], "vol_scores": []}
        for modality in itertools.chain(opt.modality, ["ensemble"])
    }
    roc_auc_elements["labels"] = []

    for i, data in metric_logger.log_every(
        dataloader, print_freq=opt.print_freq, header=f"[{suffix} {epoch}]"
    ):
        if (opt.debug or opt.wholetest) and i > 50:
            break

        for modality, optimizer in optimizer_dict.items():
            optimizer.zero_grad()

        image = data["image"].to(opt.device)
        unnormalized_image = data["unnormalized_image"].to(opt.device)
        label = data["label"].to(opt.device)
        mask = data["mask"].to(opt.device)
        spixel = data["spixel"].to(opt.device) if opt.mvc_spixel else None

        outputs = model(
            image,
            seg_size=None
            if opt.loss_on_mid_map
            else [image.shape[-2], image.shape[-1]],
        )

        losses = criterion(
            outputs,
            label,
            mask,
            epoch=epoch,
            max_epoch=opt.epochs,
            spixel=spixel,
            raw_image=unnormalized_image,
        )
        total_loss = losses["total_loss"]
        total_loss.backward()

        for modality in opt.modality:
            if opt.grad_clip > 0.0:
                grad_norm = nn.utils.clip_grad_norm_(
                    model.sub_models[modality].parameters(), opt.grad_clip
                )
                metric_logger.update(**{f"grad_norm/{modality}": grad_norm})

            optimizer_dict[modality].step()

        # image-level metrices logger
        roc_auc_elements["labels"].extend(label.tolist())
        for modality in itertools.chain(opt.modality, ["ensemble"]):
            roc_auc_elements[modality]["map_scores"].extend(
                outputs[modality]["map_pred"].tolist()
            )
            roc_auc_elements[modality]["vol_scores"].extend(
                (outputs[modality]["vol_pred"]).tolist()
            )

        metric_logger.update(**losses)

    image_metrics = update_image_roc_auc_metric(
        opt.modality + ["ensemble"], roc_auc_elements, None
    )
    metric_logger.update(**image_metrics)

    metric_logger.write_tensorboard(epoch)
    print("Average status:")
    print(metric_logger.stat_table())


def bundled_evaluate(
    model: nn.Module, dataloaders: Dict, criterion, epoch, writer, suffix, opt
):

    metric_logger = misc.MetricLogger(writer=writer, suffix=suffix + "_avg")
    for dataset, dataloader in dataloaders.items():
        outputs = evaluate(
            model,
            dataloader,
            criterion,
            dataset,
            epoch,
            writer,
            suffix + f"_{dataset}",
            opt,
        )
        old_keys = list(outputs.keys())
        for k in old_keys:
            outputs[k.replace(dataset.upper(), "AVG")] = outputs[k]
        for k in old_keys:
            del outputs[k]

        metric_logger.update(**outputs)

    metric_logger.write_tensorboard(epoch)
    print("Average status:")
    print(metric_logger.stat_table())
    return metric_logger.get_meters()


def evaluate(
    model: nn.Module,
    dataloader,
    criterion,
    dataset_title: str,
    epoch: int,
    writer,
    suffix: str,
    opt,
):

    metric_logger = misc.MetricLogger(writer=writer, suffix=suffix)
    cprint("{}-th epoch evaluation on {}".format(epoch, dataset_title.upper()), "blue")

    model.eval()

    if opt.crf_postproc:
        postprocess = DenseCRF(
            iter_max=opt.crf_iter_max,
            pos_w=opt.crf_pos_w,
            pos_xy_std=opt.crf_pos_xy_std,
            bi_w=opt.crf_bi_w,
            bi_xy_std=opt.crf_bi_xy_std,
            bi_rgb_std=opt.crf_bi_rgb_std,
        )
    elif opt.convcrf_postproc:
        convcrf_config = convcrf.default_conf
        convcrf_config["skip_init_softmax"] = True
        convcrf_config["final_softmax"] = True
        shape = [opt.convcrf_shape, opt.convcrf_shape]
        postprocess = convcrf.GaussCRF(
            conf=convcrf_config, shape=shape, nclasses=2, use_gpu=True
        ).to(opt.device)

    figure_path = opt.figure_path + f"_{dataset_title.upper()}"
    if opt.save_figure:
        if os.path.exists(figure_path):
            shutil.rmtree(figure_path)
        os.mkdir(figure_path)
        cprint("Saving figures to {}".format(figure_path), "blue")

    if opt.max_pool_postproc > 1:
        max_pool = nn.MaxPool2d(
            kernel_size=opt.max_pool_postproc,
            stride=1,
            padding=(opt.max_pool_postproc - 1) // 2,
        ).to(opt.device)
    else:
        max_pool = nn.Identity().to(opt.device)
    # used_sliding_prediction = False
    roc_auc_elements = {
        modality: {"map_scores": [], "vol_scores": []}
        for modality in itertools.chain(opt.modality, ["ensemble"])
    }
    roc_auc_elements["labels"] = []
    with torch.no_grad():
        for i, data in metric_logger.log_every(
            dataloader, print_freq=opt.print_freq, header=f"[{suffix} {epoch}]"
        ):
            if (opt.debug or opt.wholetest) and i > 50:
                break

            image_size = data["image"].shape[-2:]
            label = data["label"]
            mask = data["mask"]
            if opt.crf_postproc or opt.spixel_postproc or opt.convcrf_postproc:
                spixel = data["spixel"].to(opt.device)
            if max(image_size) > opt.tile_size and opt.large_image_strategy == "slide":
                outputs = sliding_predict(
                    model, data, opt.tile_size, opt.tile_overlap, opt
                )
            else:
                image = data["image"].to(opt.device)
                outputs = model(image, seg_size=image.shape[-2:])

            if opt.max_pool_postproc > 1:
                for modality in itertools.chain(opt.modality, ["ensemble"]):
                    outputs[modality]["out_map"] = max_pool(
                        outputs[modality]["out_map"]
                    )
            # CRF
            if opt.crf_postproc:
                raw_prob = outputs["ensemble"]["out_map"]
                image = data["unnormalized_image"] * 255.0
                if opt.crf_downsample > 1:
                    image = (
                        torch.nn.functional.interpolate(
                            image,
                            size=(
                                image_size[0] // opt.crf_downsample,
                                image_size[1] // opt.crf_downsample,
                            ),
                            mode="bilinear",
                            align_corners=False,
                        )
                        .clamp(0, 255)
                        .int()
                    )
                image = image.squeeze(0).numpy().astype(np.uint8).transpose(1, 2, 0)
                for modality in itertools.chain(opt.modality, ["ensemble"]):
                    prob = outputs[modality]["out_map"].squeeze(1)
                    if opt.crf_downsample > 1:
                        prob = (
                            torch.nn.functional.interpolate(
                                prob,
                                size=(
                                    image_size[0] // opt.crf_downsample,
                                    image_size[1] // opt.crf_downsample,
                                ),
                                mode="bilinear",
                                align_corners=False,
                            )
                            .clamp(0, 1)
                            .squeeze(0)
                        )
                    prob = torch.cat([prob, 1 - prob], dim=0).detach().cpu().numpy()
                    prob = postprocess(image, prob)
                    prob = prob[None, 0, ...]
                    prob = torch.tensor(prob, device=opt.device).unsqueeze(0)
                    if opt.crf_downsample > 1:
                        prob = torch.nn.functional.interpolate(
                            prob, size=image_size, mode="bilinear", align_corners=False
                        ).clamp(0, 1)
                    outputs[modality]["out_map"] = prob
                    outputs[modality]["map_pred"] = (
                        outputs[modality]["out_map"].max().unsqueeze(0)
                    )
            elif opt.convcrf_postproc:
                raw_prob = outputs["ensemble"]["out_map"]
                image = data["unnormalized_image"].to(opt.device) * 255.0
                image = (
                    torch.nn.functional.interpolate(
                        image,
                        size=(opt.convcrf_shape, opt.convcrf_shape),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .clamp(0, 255)
                    .int()
                )
                for modality in itertools.chain(opt.modality, ["ensemble"]):
                    prob = outputs[modality]["out_map"]
                    prob = torch.cat([prob, 1 - prob], dim=1)
                    prob = torch.nn.functional.interpolate(
                        prob,
                        size=(opt.convcrf_shape, opt.convcrf_shape),
                        mode="bilinear",
                        align_corners=False,
                    ).clamp(0, 1)
                    prob = postprocess(unary=prob, img=image)
                    prob = torch.nn.functional.interpolate(
                        prob, size=image_size, mode="bilinear", align_corners=False
                    ).clamp(0, 1)
                    outputs[modality]["out_map"] = prob[:, 0, None, ...]
                    outputs[modality]["map_pred"] = (
                        outputs[modality]["out_map"].max().unsqueeze(0)
                    )
            elif opt.spixel_postproc:
                raw_prob = outputs["ensemble"]["out_map"]
                for modality in itertools.chain(opt.modality, ["ensemble"]):
                    outputs[modality]["out_map"] = get_spixel_tgt_map(
                        outputs[modality]["out_map"], spixel
                    )

            # image-level metrices logger
            roc_auc_elements["labels"].extend(label.detach().cpu().tolist())
            for modality in itertools.chain(opt.modality, ["ensemble"]):
                roc_auc_elements[modality]["map_scores"].extend(
                    outputs[modality]["map_pred"].detach().cpu().tolist()
                )
                roc_auc_elements[modality]["vol_scores"].extend(
                    (outputs[modality]["vol_pred"]).detach().cpu().tolist()
                )

            # generate binary prediction mask
            out_map = {
                modality: outputs[modality]["out_map"] > opt.mask_threshold
                for modality in itertools.chain(opt.modality, ["ensemble"])
            }

            # only compute pixel-level metrics for manipulated images
            if label.item() == 1.0:
                for modality in itertools.chain(opt.modality, ["ensemble"]):
                    pixel_metrics = misc.calculate_pixel_f1(
                        out_map[modality].float().detach().cpu().numpy().flatten(),
                        mask.detach().cpu().numpy().flatten(),
                        suffix=f"/{modality}",
                    )
                    metric_logger.update(**pixel_metrics)

            # save images, mask, and prediction map
            if opt.save_figure:
                unnormalized_image = data["unnormalized_image"]
                # image_id = data['id'][0].split('.')[0]
                image_id = Path(data["id"][0]).stem
                save_image(
                    (
                        outputs["ensemble"]["out_map"][0, ...] > opt.mask_threshold
                    ).float()
                    * 255,
                    os.path.join(figure_path, f"{image_id}_ensemble_map.png"),
                )

    image_metrics = update_image_roc_auc_metric(
        opt.modality + ["ensemble"],
        roc_auc_elements,
        {
            modality: metric_logger.meters[f"pixel_f1/{modality}"].avg
            for modality in itertools.chain(opt.modality, ["ensemble"])
        },
    )
    metric_logger.update(**image_metrics)

    metric_logger.prepend_subprefix(f"{dataset_title.upper()}_")
    metric_logger.write_tensorboard(epoch)
    print("Average status:")
    print(metric_logger.stat_table())

    return metric_logger.get_meters()


def update_image_roc_auc_metric(modalities: List, roc_auc_elements, pixel_f1=None):

    result = {}
    for modality in modalities:
        image_metrics = misc.calculate_img_score(
            np.array(roc_auc_elements[modality]["map_scores"]) > 0.5,
            (np.array(roc_auc_elements["labels"]) > 0).astype(np.int),
            suffix=f"/{modality}",
        )
        if pixel_f1 is not None:
            image_f1 = image_metrics[f"image_f1/{modality}"]
            combined_f1 = hmean([image_f1, pixel_f1[modality]])
            image_metrics[f"comb_f1/{modality}"] = float(combined_f1)
        if 0.0 in roc_auc_elements["labels"] and 1.0 in roc_auc_elements["labels"]:
            image_auc = metrics.roc_auc_score(
                roc_auc_elements["labels"], roc_auc_elements[modality]["map_scores"]
            )
            image_metrics[f"image_auc/{modality}"] = image_auc
        result.update(image_metrics)

    return result


def pad_image(image, target_size):
    image_size = image.shape[-2:]
    if image_size != target_size:
        row_missing = target_size[0] - image_size[0]
        col_missing = target_size[1] - image_size[1]
        image = nn.functional.pad(
            image, (0, row_missing, 0, col_missing), "constant", 0
        )
    return image


def sliding_predict(model: nn.Module, data, tile_size, tile_overlap, opt):
    image = data["image"]
    mask = data["mask"]
    image = image.to(opt.device)
    image_size = image.shape[-2:]
    stride = ceil(tile_size * (1 - tile_overlap))
    tile_rows = int(ceil((image_size[0] - tile_size) / stride) + 1)
    tile_cols = int(ceil((image_size[1] - tile_size) / stride) + 1)
    result = {}
    for modality in itertools.chain(opt.modality, ["ensemble"]):
        result[modality] = {
            "out_map": torch.zeros_like(
                mask, requires_grad=False, dtype=torch.float32, device=opt.device
            ),
            "out_vol_map": torch.zeros_like(
                mask, requires_grad=False, dtype=torch.float32, device=opt.device
            ),
        }
    map_counter = torch.zeros_like(
        mask, requires_grad=False, dtype=torch.float32, device=opt.device
    )

    with torch.no_grad():
        for row in range(tile_rows):
            for col in range(tile_cols):
                x1 = int(col * stride)
                y1 = int(row * stride)
                x2 = min(x1 + tile_size, image_size[1])
                y2 = min(y1 + tile_size, image_size[0])
                x1 = max(int(x2 - tile_size), 0)
                y1 = max(int(y2 - tile_size), 0)

                image_tile = image[:, :, y1:y2, x1:x2]
                image_tile = pad_image(image_tile, [opt.tile_size, opt.tile_size])
                tile_outputs = model(image_tile, seg_size=(image_tile.shape[-2:]))
                for modality in itertools.chain(opt.modality, ["ensemble"]):
                    result[modality]["out_map"][:, :, y1:y2, x1:x2] += tile_outputs[
                        modality
                    ]["out_map"][:, :, : y2 - y1, : x2 - x1]
                    out_vol_map = get_volume_seg_map(
                        tile_outputs[modality]["out_vol"],
                        size=image_tile.shape[-2:],
                        label=data["label"],
                        kmeans=KMeans(2) if opt.consistency_kmeans else None,
                    )[:, :, : y2 - y1, : x2 - x1]
                    result[modality]["out_vol_map"][:, :, y1:y2, x1:x2] += out_vol_map
                map_counter[:, :, y1:y2, x1:x2] += 1

        for modality in itertools.chain(opt.modality, ["ensemble"]):
            result[modality]["out_map"] /= map_counter
            result[modality]["out_vol_map"] /= map_counter
            result[modality]["map_pred"] = (
                result[modality]["out_map"].max().unsqueeze(0)
            )
            result[modality]["vol_pred"] = (
                result[modality]["out_vol_map"].max().unsqueeze(0)
            )

    return result
