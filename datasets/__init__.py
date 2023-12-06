from typing import Dict

import albumentations as A

from .dataset import ImageDataset, crop_to_smallest_collate_fn


def get_dataset(datalist: Dict, subset, transform, opt):
    datasets = {}
    for k, v in datalist.items():
        # val_transform = transform
        if k in ["imd2020", "nist16"]:
            val_transform = A.Compose([A.SmallestMaxSize(opt.tile_size)])
        else:
            val_transform = transform
        datasets[k] = ImageDataset(
            k,
            v,
            subset,
            val_transform,
            opt.uncorrect_label,
            opt.mvc_spixel
            if subset == "train"
            else opt.crf_postproc or opt.convcrf_postproc or opt.spixel_postproc,
            opt.mvc_num_spixel,
        )

    return datasets
