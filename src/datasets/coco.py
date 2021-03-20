import os
import json
from collections import Counter

# installed
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# local
from utils import change_box_order


def load_coco_json(path):
    """Download json with annotations.

    Args:
        path (str): path to .json file

    Returns:
        images mapping and categories mapping
    """

    with open(path, "r") as in_file:
        content = json.load(in_file)

    images = {}  # image_id -> {file_name, height, width, annotations([{id, iscrowd, category_id, bbox}, ...])}
    for record in content["images"]:
        images[record["id"]] = {
            "file_name": record["file_name"],
            "height": record["height"],
            "width": record["width"],
            "annotations": [],
        }

    categories = {}  # category_id -> name
    for record in content["categories"]:
        categories[record["id"]] = record["name"]

    for record in content["annotations"]:
        images[record["image_id"]]["annotations"].append(
            {
                "id": record["id"],
                "iscrowd": record["iscrowd"],
                "category_id": record["category_id"],
                "bbox": record["bbox"],
            }
        )

    return images, categories


class COCOFileDataset(Dataset):
    def __init__(self, file, img_dir=None, num_anchors=8732, transforms=None, background_id=None):
        self.file = file
        self.img_dir = img_dir
        self.num_anchors = num_anchors
        self.transforms = transforms
        self.background_id = background_id

        self.images, self.categories = load_coco_json(file)
        self.images_list = sorted(self.images.keys())

        if background_id is None:
            self.background_cls = 0
            self.class_to_cid = {self.background_cls: -1}
            for cls_idx, cat_id in enumerate(sorted(self.categories.keys()), start=1):
                self.class_to_cid[cls_idx] = cat_id
            self.cid_to_class = {v: k for k, v in self.class_to_cid.items()}
        else:
            self.class_to_cid = {cls_idx: cat_id for cls_idx, cat_id in enumerate(sorted(self.categories.keys()))}
            self.cid_to_class = {v: k for k, v in self.class_to_cid.items()}
            self.background_cls = self.cid_to_class[background_id]

    def __len__(self):
        return len(self.images_list)

    def info(self):
        """Information about dataset.

        Returns:
            str with information about dataset
        """
        txt = (
            "               Num images: {}\n"
            "              Num classes: {} (with background)\n"
            "            Background id: {}\n"
            "         Background class: {}\n"
            "               Categories: {}\n"
            "   Num images with bboxes: {}\n"
            "Num images without bboxes: {}\n"
            "          BBox statistics: {}"
        )
        txt = txt.format(
            len(self.images),
            len(self.categories),
            "not specified" if self.background_id is None else self.background_id,
            self.background_cls,
            sorted(self.categories),
            sum(1 for img in self.images.values() if len(img["annotations"]) != 0),
            sum(1 for img in self.images.values() if len(img["annotations"]) == 0),
            dict(
                Counter(
                    self.categories[annot["category_id"]]
                    for img in self.images.values()
                    for annot in img["annotations"]
                ).most_common()
            ),
        )
        return txt

    @staticmethod
    def _read_image(path):
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def _from_pixels_to_pcnt(box, width, height):
        x, y, w, h = box
        # fmt: off
        return [
            x / width,
            y / height,
            (x + w) / width,
            (y + h) / height
        ]
        # fmt: on

    def __getitem__(self, index):
        img_id = self.images_list[index]
        img_record = self.images[img_id]

        path = img_record["file_name"]
        if self.img_dir is not None:
            path = os.path.join(self.img_dir, path)
        image = self._read_image(path)

        boxes = []  # each element is a tuple of (x1, y1, x2, y2, "class")
        for annotation in img_record["annotations"]:
            xyxy = self._from_pixels_to_pcnt(
                annotation["bbox"],
                img_record["width"],
                img_record["height"],
            )
            assert all(0 <= num <= 1 for num in xyxy), f"All numbers should be in range [0, 1], but got {xyxy}!"
            bbox_class = str(self.cid_to_class[annotation["category_id"]])
            boxes.append(xyxy + [str(bbox_class)])

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)
            image, boxes = transformed["image"], transformed["bboxes"]

        bboxes = np.zeros((self.num_anchors, 4), dtype=np.float32)
        classes = np.full(self.num_anchors, self.background_cls, dtype=np.int32)
        for idx, (x1, y1, x2, y2, box_cls) in enumerate(boxes):
            bboxes[idx, :] = [x1, y1, x2, y2]
            classes[idx] = int(box_cls)

        bboxes = torch.from_numpy(bboxes)
        bboxes = change_box_order(bboxes, "xyxy2xywh")
        classes = torch.LongTensor(classes)

        return image, bboxes, classes
