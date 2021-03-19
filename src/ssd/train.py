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
from ssd.model import SSD300, Loss
from datasets import COCOFileDataset
from utils import t2d, seed_all, get_logger, export_to_onnx


logger = get_logger("ssd300")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_fn(loader, model, device, criterion, optimizer, scheduler=None, verbose=True):
    model.train()
    num_batches = len(loader)
    metrics = {"regression_loss": 0.0, "classification_loss": 0.0, "loss": 0.0}
    progress_str = "regression loss {:.4f}, classification loss {:.4f}, loss {:.4f}"
    with tqdm(desc="     train", total=num_batches, file=sys.stdout, disable=not verbose) as progress:
        for batch_idx, batch in enumerate(loader):
            imgs, boxes, classes = t2d(batch, device)

            locs, confs = model(imgs)

            regression_loss, classification_loss = criterion(locs, boxes, confs, classes)
            loss = regression_loss + classification_loss

            optimizer.zero_grad()
            loss.backward()

            _rloss = regression_loss.item()
            _closs = classification_loss.item()
            _loss = loss.item()
            metrics["regression_loss"] += _rloss
            metrics["classification_loss"] += _closs
            metrics["loss"] += _loss

            progress.set_postfix_str(progress_str.format(_rloss, _closs, _loss))

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            progress.update(1)
    metrics = {k: (v / num_batches) for k, v in metrics.items()}
    metrics["lr"] = get_lr(optimizer)
    return metrics


@torch.no_grad()
def valid_fn(loader, model, device, criterion, verbose=True):
    model.eval()
    num_batches = len(loader)
    metrics = {"regression_loss": 0.0, "classification_loss": 0.0, "loss": 0.0}
    with tqdm(desc="validation", total=num_batches, file=sys.stdout, disable=not verbose) as progress:
        for batch_idx, batch in enumerate(loader):
            imgs, boxes, classes = t2d(batch, device)

            locs, confs = model(imgs)

            regression_loss, classification_loss = criterion(locs, boxes, confs, classes)
            loss = regression_loss + classification_loss

            _rloss = regression_loss.item()
            _closs = classification_loss.item()
            _loss = loss.item()
            metrics["regression_loss"] += _rloss
            metrics["classification_loss"] += _closs
            metrics["loss"] += _loss

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
            albu.Resize(300, 300),
            albu.Normalize(),
            ToTensorV2(),
        ],
        bbox_params=albu.BboxParams("albumentations"),  # 'albumentations' because x1, y1, x2, y2 in range [0, 1]
    )
    train_dataset = COCOFileDataset(
        train_config["annotations"], train_config["images_dir"], transforms=train_augmentations
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        num_workers=train_config["num_workers"],
        shuffle=True,
        drop_last=True,
    )
    logger.info("Train dataset information:")
    logger.info("\n" + train_dataset.info())

    valid_config = args["validation"]
    valid_augmentations = albu.Compose(
        [
            albu.Resize(300, 300),
            albu.Normalize(),
            ToTensorV2(),
        ],
        bbox_params=albu.BboxParams(format="albumentations"),  # 'albumentations' because x1, y1, x2, y2 in range [0, 1]
    )
    valid_dataset = COCOFileDataset(
        valid_config["annotations"], valid_config["images_dir"], transforms=valid_augmentations
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=valid_config["batch_size"],
        num_workers=valid_config["num_workers"],
        shuffle=False,
        drop_last=False,
    )
    logger.info("Validation dataset information:")
    logger.info("\n" + valid_dataset.info())

    model_config = args["model"]
    num_classes = model_config["num_classes"] + 1  # +1 for background class
    model = SSD300(model_config["backbone"], num_classes)
    model = model.to(device)
    #     optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.SGD(model.parameters(), lr=2.6e-3, momentum=0.9, weight_decay=0.0005)
    epoch_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args["experiment"]["num_epochs"])
    batch_scheduler = None
    criterion = Loss(num_classes)

    experiment_config = args["experiment"]
    num_epochs = experiment_config["num_epochs"]
    for epoch_idx in range(1, num_epochs + 1):
        logger.info(f"Epoch: {epoch_idx}/{num_epochs}")
        train_metrics = train_fn(train_loader, model, device, criterion, optimizer, batch_scheduler, verbose=False)
        logger.info(f"     Train: {train_metrics}")

        # TODO: checkpoints
        valid_metrics = valid_fn(valid_loader, model, device, criterion, verbose=False)
        logger.info(f"Validation: {valid_metrics}")

        epoch_scheduler.step()

    export_to_onnx(model, torch.randn(1, 3, 300, 300), experiment_config["onnx"])
    logger.info("Exported ONNX model to '{}'".format(experiment_config["onnx"]))


def main():
    config = {
        "train": {
            "annotations": "data/dataset.json",
            "images_dir": "data/images",
            "batch_size": 2,
            "num_workers": 2,
        },
        "validation": {
            "annotations": "data/dataset.json",
            "images_dir": "data/images",
            "batch_size": 2,
            "num_workers": 2,
        },
        "model": {
            "backbone": "resnet18",
            "num_classes": 81,
        },
        "experiment": {
            "num_epochs": 50,
            "onnx": "ssd300.onnx",
        },
    }
    experiment("cuda:0", config)


if __name__ == "__main__":
    main()