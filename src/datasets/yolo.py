import os

import numpy as np
import torch

from utils import change_box_order
from .coco import COCOFileDataset


class YoLoFileDataset(COCOFileDataset):
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

        n_boxes = len(boxes)
        bboxes = np.zeros((n_boxes, 4), dtype=np.float32)
        classes = np.zeros(n_boxes, dtype=np.int32)
        for idx, (x1, y1, x2, y2, box_cls) in enumerate(boxes):
            bboxes[idx, :] = [x1, y1, x2, y2]
            classes[idx] = int(box_cls)

        bboxes = torch.from_numpy(bboxes)
        bboxes = change_box_order(bboxes, "xyxy2xywh")
        classes = torch.LongTensor(classes)

        return image, bboxes, classes

    @staticmethod
    def collate_fn(batch):
        """Collect batch for yolo format.

        Args:
            batch (List[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]]):
                List with records from YoLoFileDataset.

        Returns:
            images batch with shape [B, C, H, W]
            sample indices with shape [sum(num_bboxes_i for each bbox in batch), ]
            boxes with shape [sum(num_bboxes_i for each bbox in batch), 4]
            classes with shape [sum(num_bboxes_i for each bbox in batch), ]
        """
        images, boxes, classes = list(zip(*batch))

        sample_indices = []
        for i, box in enumerate(boxes):
            sample_indices += [i] * box.shape[0]

        images = torch.stack(images)
        sample_indices = torch.Tensor(sample_indices)
        boxes = torch.cat(boxes, 0)
        classes = torch.cat(classes, 0)

        return images, sample_indices, boxes, classes
