import sys
import logging

# installed
from tqdm import tqdm
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# local
from yolov3.model import yolov3_tiny
from datasets import YoLoFileDataset
from utils import t2d, seed_all, get_logger, export_to_onnx


logger = get_logger("yolov3")


def _get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_fn(loader, model, device, optimizer, scheduler=None, verbose=True):
    model.train()
    num_batches = len(loader)
    metrics = {"loss": 0.0}
    for idx, _ in enumerate(model.yolo_layers):
        metrics[f"yolo{idx}_cls_acc"] = 0.0
        metrics[f"yolo{idx}_recall50"] = 0.0
        metrics[f"yolo{idx}_recall75"] = 0.0
        metrics[f"yolo{idx}_precision"] = 0.0
        metrics[f"yolo{idx}_conf_obj"] = 0.0
        metrics[f"yolo{idx}_conf_no_obj"] = 0.0
    progress_str = "loss {:.4f}"
    with tqdm(desc="     train", total=num_batches, file=sys.stdout, disable=not verbose) as progress:
        for batch_idx, batch in enumerate(loader):
            imgs, sample_idxs, boxes, classes = batch
            imgs, tgts = t2d(
                (
                    imgs,
                    torch.cat([sample_idxs.unsqueeze(-1), classes.unsqueeze(-1), boxes], -1),
                ),
                device,
            )
            # imgs have shape [BATCH, C, W, H]
            # tgts have shape [BATCH_NUM_BOXES, 1 + 1 + 4]

            loss, _ = model(imgs, tgts)

            optimizer.zero_grad()
            loss.backward()

            _loss = loss.item()
            metrics["loss"] += _loss

            for idx, layer in enumerate(model.yolo_layers):
                metrics[f"yolo{idx}_cls_acc"] = layer.metrics["cls_acc"]
                metrics[f"yolo{idx}_recall50"] = layer.metrics["recall50"]
                metrics[f"yolo{idx}_recall75"] = layer.metrics["recall75"]
                metrics[f"yolo{idx}_precision"] = layer.metrics["precision"]
                metrics[f"yolo{idx}_conf_obj"] = layer.metrics["conf_obj"]
                metrics[f"yolo{idx}_conf_no_obj"] = layer.metrics["conf_no_obj"]

            progress.set_postfix_str(progress_str.format(_loss))

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            progress.update(1)
    metrics = {k: (v / num_batches) for k, v in metrics.items()}
    metrics["lr"] = _get_lr(optimizer)
    return metrics


@torch.no_grad()
def valid_fn(loader, model, device, verbose=True):
    model.eval()
    num_batches = len(loader)
    metrics = {"loss": 0.0}
    for idx, _ in enumerate(model.yolo_layers):
        metrics[f"yolo{idx}_cls_acc"] = 0.0
        metrics[f"yolo{idx}_recall50"] = 0.0
        metrics[f"yolo{idx}_recall75"] = 0.0
        metrics[f"yolo{idx}_precision"] = 0.0
        metrics[f"yolo{idx}_conf_obj"] = 0.0
        metrics[f"yolo{idx}_conf_no_obj"] = 0.0
    with tqdm(desc="validation", total=num_batches, file=sys.stdout, disable=not verbose) as progress:
        for batch_idx, batch in enumerate(loader):
            imgs, sample_idxs, boxes, classes = batch
            imgs, tgts = t2d(
                (
                    imgs,
                    torch.cat([sample_idxs.unsqueeze(-1), classes.unsqueeze(-1), boxes], -1),
                ),
                device,
            )

            loss, _ = model(imgs, tgts)

            _loss = loss.item()
            metrics["loss"] += _loss

            for idx, layer in enumerate(model.yolo_layers):
                metrics[f"yolo{idx}_cls_acc"] = layer.metrics["cls_acc"]
                metrics[f"yolo{idx}_recall50"] = layer.metrics["recall50"]
                metrics[f"yolo{idx}_recall75"] = layer.metrics["recall75"]
                metrics[f"yolo{idx}_precision"] = layer.metrics["precision"]
                metrics[f"yolo{idx}_conf_obj"] = layer.metrics["conf_obj"]
                metrics[f"yolo{idx}_conf_no_obj"] = layer.metrics["conf_no_obj"]

            progress.update(1)
    metrics = {k: (v / num_batches) for k, v in metrics.items()}
    return metrics


def experiment(device, args=None):
    """Train model.

    Args:
        device (str): device to use for training.
        args (dict): experiment arguments.
    """
    if args is None:
        args = dict

    train_config = args["train"]
    train_augmentations = albu.Compose(
        [
            albu.OneOf(
                [
                    albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25),
                    albu.RandomGamma(),
                    albu.CLAHE(),
                ]
            ),
            albu.RandomBrightnessContrast(brightness_limit=[-0.3, 0.3], contrast_limit=[-0.3, 0.3], p=0.5),
            albu.OneOf([albu.Blur(), albu.MotionBlur(), albu.GaussNoise(), albu.ImageCompression(quality_lower=75)]),
            albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=10, border_mode=0, p=0.5),
            albu.Resize(416, 416),
            albu.Normalize(),
            ToTensorV2(),
        ],
        bbox_params=albu.BboxParams("albumentations"),  # 'albumentations' because x1, y1, x2, y2 in range [0, 1]
    )
    train_dataset = YoLoFileDataset(
        train_config["annotations"], train_config["images_dir"], transforms=train_augmentations
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        num_workers=train_config["num_workers"],
        shuffle=True,
        drop_last=(len(train_dataset) > train_config["batch_size"]),
        collate_fn=YoLoFileDataset.collate_fn,
    )
    logger.info("Train dataset information:\n" + train_dataset.info())

    valid_config = args["validation"]
    valid_augmentations = albu.Compose(
        [
            albu.Resize(416, 416),
            albu.Normalize(),
            ToTensorV2(),
        ],
        bbox_params=albu.BboxParams(format="albumentations"),  # 'albumentations' because x1, y1, x2, y2 in range [0, 1]
    )
    valid_dataset = YoLoFileDataset(
        valid_config["annotations"], valid_config["images_dir"], transforms=valid_augmentations
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=valid_config["batch_size"],
        num_workers=valid_config["num_workers"],
        shuffle=False,
        drop_last=False,
        collate_fn=YoLoFileDataset.collate_fn,
    )
    logger.info("Validation dataset information:\n" + valid_dataset.info())

    model_config = args["model"]
    num_classes = model_config["num_classes"] + 1  # +1 for background class
    seed_all(42)
    model = yolov3_tiny(num_classes=num_classes)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=2.6e-3, momentum=0.9, weight_decay=0.0005)
    epoch_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args["experiment"]["num_epochs"])
    batch_scheduler = None

    experiment_config = args["experiment"]
    num_epochs = experiment_config["num_epochs"]
    for epoch_idx in range(1, num_epochs + 1):
        logger.info(f"Epoch: {epoch_idx}/{num_epochs}")
        train_metrics = train_fn(train_loader, model, device, optimizer, batch_scheduler, verbose=False)
        logger.info(f"     Train: {train_metrics}")

        # TODO: checkpoints
        valid_metrics = valid_fn(valid_loader, model, device, verbose=False)
        logger.info(f"Validation: {valid_metrics}")

        epoch_scheduler.step()

    torch.onnx.export(
        model.eval(),
        torch.randn(1, 3, 416, 416).to(device),
        experiment_config["onnx"],
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["preds"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "locs": {0: "batch_size"},
            "confs": {0: "batch_size"},
        },
    )
    logger.info("Exported ONNX model to '{}'".format(experiment_config["onnx"]))


def main():
    config = {
        "train": {
            "annotations": "data/dataset.json",
            "images_dir": "data/images",
            "batch_size": 4,
            "num_workers": 4,
        },
        "validation": {
            "annotations": "data/dataset.json",
            "images_dir": "data/images",
            "batch_size": 4,
            "num_workers": 4,
        },
        "model": {
            "backbone": "resnet18",
            "num_classes": 81,
        },
        "experiment": {
            "num_epochs": 10,
            "onnx": "yolov3-tiny.onnx",
        },
    }
    experiment("cuda:0", config)


if __name__ == "__main__":
    main()