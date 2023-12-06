import datetime
import math
import os
from functools import partial

import albumentations as A
import torch.optim as optim
from termcolor import cprint
from timm.scheduler import create_scheduler
from torch.utils.data import DataLoader

import utils.misc as misc
from datasets import crop_to_smallest_collate_fn, get_dataset
from engine import bundled_evaluate, train
from losses import get_bundled_loss, get_loss
from models import get_ensemble_model, get_single_modal_model
from opt import get_opt


def main(opt):
    # get tensorboard writer
    writer = misc.setup_env(opt)

    # dataset
    # training sets
    train_loaders = {}
    if not opt.eval:
        train_transform = A.Compose(
            [
                A.HorizontalFlip(0.5),
                A.SmallestMaxSize(int(opt.input_size * 1.5))
                if opt.resize_aug
                else A.NoOp(),
                A.RandomSizedCrop(
                    (opt.input_size, int(opt.input_size * 1.5)),
                    opt.input_size,
                    opt.input_size,
                )
                if opt.resize_aug
                else A.NoOp(),
                A.NoOp() if opt.no_gaussian_blur else A.GaussianBlur(p=0.5),
                A.NoOp() if opt.no_color_jitter else A.ColorJitter(p=0.5),
                A.NoOp() if opt.no_jpeg_compression else A.ImageCompression(p=0.5),
            ]
        )
        train_sets = get_dataset(opt.train_datalist, "train", train_transform, opt)
        for k, dataset in train_sets.items():
            train_loaders[k] = DataLoader(
                dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=0 if opt.debug else opt.num_workers,
                collate_fn=partial(
                    crop_to_smallest_collate_fn,
                    max_size=opt.input_size,
                    uncorrect_label=opt.uncorrect_label,
                ),
            )
    # validation sets
    if opt.large_image_strategy == "rescale":
        val_transform = A.Compose([A.SmallestMaxSize(opt.tile_size)])
    else:
        val_transform = None
    val_sets = get_dataset(opt.val_datalist, opt.val_set, val_transform, opt)
    val_loaders = {}
    for k, dataset in val_sets.items():
        val_loaders[k] = DataLoader(
            dataset,
            batch_size=1,
            shuffle=opt.val_shuffle,
            pin_memory=True,
            num_workers=0 if opt.debug else opt.num_workers,
        )

    # multi-view models and optimizers
    optimizer_dict = {}
    scheduler_dict = {}
    model = get_ensemble_model(opt).to(opt.device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Number of total params: {n_param}, num params per model: {int(n_param / len(opt.modality))}"
    )

    # optimizer and scheduler
    for modality in opt.modality:
        if opt.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                model.sub_models[modality].parameters(),
                opt.lr,
                weight_decay=opt.weight_decay,
            )
        elif opt.optimizer.lower() == "sgd":
            optimizer = optim.SGD(
                model.sub_models[modality].parameters(),
                opt.lr,
                opt.momentum,
                weight_decay=opt.weight_decay,
            )
        else:
            raise RuntimeError(f"Unsupported optimizer {opt.optimizer}.")

        scheduler, num_epoch = create_scheduler(opt, optimizer)

        optimizer_dict[modality] = optimizer
        scheduler_dict[modality] = scheduler
    opt.epochs = num_epoch

    # loss functions
    # loss function including the multi-view consistency loss, for training
    bundled_criterion = get_bundled_loss(opt).to(opt.device)
    # loss function excluding the multi-view consistency loss, for evaluation
    single_criterion = get_loss(opt).to(opt.device)

    if opt.resume:
        misc.resume_from(model, opt.resume)

    if opt.eval:
        bundled_evaluate(
            model, val_loaders, single_criterion, 0, writer, suffix="val", opt=opt
        )
        return

    cprint("The training will last for {} epochs.".format(opt.epochs), "blue")
    best_ensemble_image_f1 = -math.inf
    for epoch in range(opt.epochs):
        for title, dataloader in train_loaders.items():
            train(
                model,
                dataloader,
                title,
                optimizer_dict,
                bundled_criterion,
                epoch,
                writer,
                suffix="train",
                opt=opt,
            )
        for sched_idx, scheduler in enumerate(scheduler_dict.values()):
            if sched_idx == 0 and writer is not None:
                writer.add_scalar("lr", scheduler._get_lr(epoch)[0], epoch)
            scheduler.step(epoch)

        if (epoch + 1) % opt.eval_freq == 0 or epoch in [opt.epochs - 1]:
            result = bundled_evaluate(
                model,
                val_loaders,
                single_criterion,
                epoch,
                writer,
                suffix="val",
                opt=opt,
            )
            misc.save_model(
                os.path.join(
                    opt.save_root_path, opt.dir_name, "checkpoint", f"{epoch}.pt"
                ),
                model,
                epoch,
                opt,
                performance=result,
            )
            if result["image_f1/AVG_ensemble"] > best_ensemble_image_f1:
                best_ensemble_image_f1 = result["image_f1/AVG_ensemble"]
                misc.save_model(
                    os.path.join(
                        opt.save_root_path, opt.dir_name, "checkpoint", "best.pt"
                    ),
                    model,
                    epoch,
                    opt,
                    performance=result,
                )
                misc.update_record(result, epoch, opt, "best_record")
            misc.update_record(result, epoch, opt, "latest_record")

    print("best performance:", best_ensemble_image_f1)


if __name__ == "__main__":
    opt = get_opt()

    # import cProfile
    # import pstats
    # profiler = cProfile.Profile()
    # profiler.enable()

    st = datetime.datetime.now()
    main(opt)
    total_time = datetime.datetime.now() - st
    total_time = str(datetime.timedelta(seconds=total_time.seconds))
    print(f"Total time: {total_time}")

    print("finished")

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.strip_dirs()
    # stats_name = f'cprofile-data{opt.suffix}'
    # if not opt.debug and not opt.eval:
    #     stats_name = os.path.join(opt.save_root_path, opt.dir_name, stats_name)
    # else:
    #     stats_name = os.path.join('tmp', stats_name)
    # stats.dump_stats(stats_name)
