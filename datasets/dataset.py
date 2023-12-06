import json
import os
import random
import signal

import albumentations as A
import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms as T
from albumentations.pytorch.functional import img_to_tensor, mask_to_tensor
from skimage import segmentation
from termcolor import cprint
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        datalist: str,
        mode: str,
        transform=None,
        uncorrect_label=False,
        spixel: bool = False,
        num_spixel: int = 100,
    ):
        super().__init__()

        assert os.path.exists(datalist), f"{datalist} does not exist"
        assert mode in ["train", "val"], f"{mode} unsupported mode"

        with open(datalist, "r") as f:
            self.datalist = json.load(f)

        self.datalist = dict(
            filter(lambda x: x[1]["subset"] == mode, self.datalist.items())
        )
        if len(self.datalist) == 0:
            raise NotImplementedError(f"no item in {datalist} {mode} dataset")
        self.video_id_list = list(self.datalist.keys())
        self.transform = transform
        self.uncorrect_label = uncorrect_label

        self.dataset_name = dataset_name
        h5_path = os.path.join("data", dataset_name + "_dataset.hdf5")
        self.use_h5 = os.path.exists(h5_path)
        if self.use_h5:
            cprint(
                f"{dataset_name} {mode} HDF5 database found, loading into memory...",
                "blue",
            )
            try:
                with timeout(seconds=60):
                    self.database = h5py.File(h5_path, "r", driver="core")
            except Exception as e:
                self.database = h5py.File(h5_path, "r")
                cprint(
                    "Failed to load {} HDF5 database to memory due to {}".format(
                        dataset_name, str(e)
                    ),
                    "red",
                )
        else:
            cprint(
                f"{dataset_name} {mode} HDF5 database not found, using raw images.",
                "blue",
            )

        self.spixel = False
        self.num_spixel = num_spixel
        if spixel:
            self.spixel = True
            self.spixel_dict = {}

    def __getitem__(self, index):
        image_id = self.video_id_list[index]
        info = self.datalist[image_id]
        label = float(info["label"])
        if self.use_h5:
            try:
                image = self.database[info["path"].replace("/", "-")][()]
            except Exception as e:
                cprint(
                    "Failed to load {} from {} due to {}".format(
                        image_id, self.dataset_name, str(e)
                    ),
                    "red",
                )
                image = cv2.imread(info["path"])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            assert os.path.exists(info["path"]), f"{info['path']} does not exist!"
            image = cv2.imread(info["path"])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.spixel and image_id not in self.spixel_dict.keys():
            spixel = segmentation.slic(
                image, n_segments=self.num_spixel, channel_axis=2, start_label=0
            )
            self.spixel_dict[image_id] = spixel

        image_size = image.shape[:2]

        # 1 means modified area, 0 means pristine
        if "mask" in info.keys():
            if self.use_h5:
                try:
                    mask = self.database[info["mask"].replace("/", "-")][()]
                except Exception as e:
                    cprint(
                        "Failed to load {} mask from {} due to {}".format(
                            image_id, self.dataset_name, str(e)
                        ),
                        "red",
                    )
                    mask = cv2.imread(info["mask"], cv2.IMREAD_GRAYSCALE)
            else:
                mask = cv2.imread(info["mask"], cv2.IMREAD_GRAYSCALE)
        else:
            if label == 0:
                mask = np.zeros(image_size)
            else:
                mask = np.ones(image_size)

        if self.transform is not None:
            if self.spixel:
                transformed = self.transform(
                    image=image, masks=[mask, self.spixel_dict[image_id]]
                )  # TODO I am not sure if this is correct for scaling
                mask = transformed["masks"][0]
                spixel = transformed["masks"][1]
            else:
                transformed = self.transform(image=image, mask=mask)
                mask = transformed["mask"]

            image = transformed["image"]
            if not self.uncorrect_label:
                label = float(mask.max() != 0.0)

        if label == 1.0 and image.shape[:-1] != mask.shape:
            mask = cv2.resize(mask, dsize=(image.shape[1], image.shape[0]))

        unnormalized_image = img_to_tensor(image)
        image = img_to_tensor(
            image,
            normalize={"mean": IMAGENET_DEFAULT_MEAN, "std": IMAGENET_DEFAULT_STD},
        )
        mask = mask_to_tensor(mask, num_classes=1, sigmoid=True)

        output = {
            "image": image,  # tensor of 3, H, W
            "label": label,  # float
            "mask": mask,  # tensor of 1, H, W
            "id": image_id,  # string
            "unnormalized_image": unnormalized_image,
        }  # tensor of 3, H, W
        if self.spixel:
            spixel = torch.from_numpy(spixel).unsqueeze(0)
            output["spixel"] = spixel
        return output

    def __len__(self):
        return len(self.video_id_list)


def crop_to_smallest_collate_fn(batch, max_size=128, uncorrect_label=False):
    # get the smallest image size in a batch
    smallest_size = [max_size, max_size]
    for item in batch:
        if item["mask"].shape[-2:] != item["image"].shape[-2:]:
            cprint(
                f"{item['id']} has inconsistent image-mask sizes,"
                "with image size {item['image'].shape[-2:]} and mask size"
                "{item['mask'].shape[-2:]}!",
                "red",
            )
        image_size = item["image"].shape[-2:]
        if image_size[0] < smallest_size[0]:
            smallest_size[0] = image_size[0]
        if image_size[1] < smallest_size[1]:
            smallest_size[1] = image_size[1]

    # crop all images and masks in each item to the smallest size
    result = {}
    for item in batch:
        image_size = item["image"].shape[-2:]
        x1 = random.randint(0, image_size[1] - smallest_size[1])
        y1 = random.randint(0, image_size[0] - smallest_size[0])
        x2 = x1 + smallest_size[1]
        y2 = y1 + smallest_size[0]
        for k in ["image", "mask", "unnormalized_image", "spixel"]:
            if k not in item.keys():
                continue
            item[k] = item[k][:, y1:y2, x1:x2]
            if not uncorrect_label:
                item["label"] = float(item["mask"].max() != 0.0)
        for k, v in item.items():
            if k in result.keys():
                result[k].append(v)
            else:
                result[k] = [v]

    # stack all outputs
    for k, v in result.items():
        if k in ["image", "mask", "unnormalized_image", "spixel"]:
            if k not in result.keys():
                continue
            result[k] = torch.stack(v, dim=0)
        elif k in ["label"]:
            result[k] = torch.tensor(v).float()

    return result


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)
