from pathlib import Path

import albumentations as A
import cv2
import torch
import tqdm
from albumentations.pytorch.functional import img_to_tensor
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.utils import draw_segmentation_masks, make_grid, save_image

import utils.misc as misc
from models import get_ensemble_model
from opt import get_opt


def demo(folder_path, output_path=Path("tmp")):
    opt = get_opt()
    model = get_ensemble_model(opt).to(opt.device)
    misc.resume_from(model, opt.resume)

    with torch.no_grad():
        for image_path in tqdm.tqdm(folder_path.glob("*.jpg")):
            image = cv2.imread(image_path.as_posix())
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dsm_image = torch.from_numpy(image).permute(2, 0, 1)
            image_size = image.shape[:2]
            raw_image = img_to_tensor(image)
            image = img_to_tensor(
                image,
                normalize={"mean": IMAGENET_DEFAULT_MEAN, "std": IMAGENET_DEFAULT_STD},
            )
            image = image.to(opt.device).unsqueeze(0)
            outputs = model(image, seg_size=image_size)
            out_map = outputs["ensemble"]["out_map"][0, ...].detach().cpu()
            pred = outputs["ensemble"]["out_map"].max().item()
            if pred > opt.mask_threshold:
                print(f"Found manipulation in {image_path.name}")
            else:
                print(f"No manipulation found in {image_path.name}")

            overlay = draw_segmentation_masks(
                dsm_image, masks=out_map[0, ...] > opt.mask_threshold
            )
            grid_image = make_grid(
                [
                    raw_image,
                    (out_map.repeat(3, 1, 1) > opt.mask_threshold).float() * 255,
                    overlay / 255.0,
                ],
                padding=5,
            )
            image_name = image_path.stem + f"-{pred:.2f}" + image_path.suffix
            save_image(grid_image, (output_path / image_name).as_posix())


if __name__ == "__main__":
    folder_path = Path("demo")  # input path
    output_path = Path("tmp")  # output path
    output_path.mkdir(exist_ok=True, parents=True)
    demo(folder_path, output_path)
