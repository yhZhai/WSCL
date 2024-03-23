from pathlib import Path

import gradio as gr
import numpy as np
import torch
from albumentations.pytorch.functional import img_to_tensor
from huggingface_hub import hf_hub_download
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.utils import draw_segmentation_masks, make_grid, save_image

import utils.misc as misc
from models import get_ensemble_model
from opt import get_opt


def greet(input_image):
    opt, model = _get_model()

    with torch.no_grad():
        image = input_image
        image = np.array(image)
        dsm_image = torch.from_numpy(image).permute(2, 0, 1)
        image_size = image.shape[:2]
        image = img_to_tensor(
            image,
            normalize={"mean": IMAGENET_DEFAULT_MEAN, "std": IMAGENET_DEFAULT_STD},
        )
        image = image.to(opt.device).unsqueeze(0)
        outputs = model(image, seg_size=image_size)
        out_map = outputs["ensemble"]["out_map"][0, ...].detach().cpu()
        pred = outputs["ensemble"]["out_map"].max().item()
        if pred > opt.mask_threshold:
            output_string = f"Found manipulation (manipulation probability {pred:.2f})."
        else:
            output_string = (
                f"No manipulation found (manipulation probability {pred:.2f})."
            )

        overlay = draw_segmentation_masks(
            dsm_image, masks=out_map[0, ...] > opt.mask_threshold
        )
        overlay = overlay.permute(1, 2, 0)
        overlay = overlay.detach().cpu().numpy()
        overlay = overlay.astype(np.uint8)
    return overlay, output_string


def _get_model(config_path="configs/final.yaml", ckpt_path="tmp/checkpoint.pt"):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        ckpt_path.parent.mkdir(exist_ok=True, parents=True)
        hf_hub_download(
            repo_id="yhzhai/WSCL",
            filename="checkpoint.pt",
            local_dir=ckpt_path.parent.as_posix(),
        )

    opt = get_opt(config_path)
    opt.resume = ckpt_path.as_posix()

    model = get_ensemble_model(opt).to(opt.device)
    misc.resume_from(model, opt.resume)
    return opt, model


iface = gr.Interface(
    fn=greet,
    title="WSCL: Image Manipulation Detection",
    inputs=gr.Image(),
    outputs=["image", "text"],
    examples=[["demo/au.jpg"], ["demo/tp.jpg"]],
    cache_examples=True,
)
iface.launch()
