import torch
import numpy as np


def change_box_order(boxes, order):
    """Change box order between
    (xmin, ymin, xmax, ymax) <-> (xcenter, ycenter, width, height).

    Args:
        boxes: (torch.Tensor or np.ndarray) bounding boxes, sized [N,4].
        order: (str) either "xyxy2xywh" or "xywh2xyxy".

    Returns:
        (torch.Tensor) converted bounding boxes, sized [N,4].
    """
    assert order in {"xyxy2xywh", "xywh2xyxy"}
    concat_fn = torch.cat if isinstance(boxes, torch.Tensor) else np.concatenate

    a = boxes[:, :2]
    b = boxes[:, 2:]
    if order == "xyxy2xywh":
        return concat_fn([(a + b) / 2, b - a], 1)
    return concat_fn([a - b / 2, a + b / 2], 1)


def box_clamp(boxes, xmin, ymin, xmax, ymax):
    """Clamp boxes.

    Args:
        boxes: (torch.Tensor) bounding boxes of (xmin, ymin, xmax, ymax), sized [N,4].
        xmin: (number) min value of x.
        ymin: (number) min value of y.
        xmax: (number) max value of x.
        ymax: (number) max value of y.

    Returns:
        clamped boxes (torch.Tensor).
    """
    boxes[:, 0].clamp_(min=xmin, max=xmax)
    boxes[:, 1].clamp_(min=ymin, max=ymax)
    boxes[:, 2].clamp_(min=xmin, max=xmax)
    boxes[:, 3].clamp_(min=ymin, max=ymax)
    return boxes


def box_select(boxes, xmin, ymin, xmax, ymax):
    """Select boxes in range (xmin, ymin, xmax, ymax).

    Args:
        boxes: (torch.Tensor) bounding boxes of (xmin, ymin, xmax, ymax), sized [N,4].
        xmin: (number) min value of x.
        ymin: (number) min value of y.
        xmax: (number) max value of x.
        ymax: (number) max value of y.

    Returns:
        selected boxes (torch.Tensor), sized [M,4].
        selected mask (torch.Tensor), sized [N,].
    """
    mask = (boxes[:, 0] >= xmin) & (boxes[:, 1] >= ymin) & (boxes[:, 2] <= xmax) & (boxes[:, 3] <= ymax)
    boxes = boxes[mask, :]
    return boxes, mask


def box_iou(box1, box2):
    """Compute the intersection over union of two set of boxes.

    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        box1: (torch.Tensor) bounding boxes, sized [N,4].
        box2: (torch.Tensor) bounding boxes, sized [M,4].

    Return:
        (torch.Tensor) iou, sized [N,M].

    Reference:
        https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def box_nms(bboxes, scores, threshold=0.5):
    """Non maximum suppression.

    Args:
        bboxes: (torch.Tensor) bounding boxes, sized [N,4].
        scores: (torch.Tensor) confidence scores, sized [N,].
        threshold: (float) overlap threshold.

    Returns:
        keep: (torch.Tensor) selected indices.

    Reference:
        https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """
    if bboxes.dim() == 1:
        bboxes = bboxes.reshape(-1, 4)

    x1, y1 = bboxes[:, 0], bboxes[:, 1]
    x2, y2 = bboxes[:, 2], bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        if order.dim() == 0:
            i = order.item()
        else:
            i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (overlap <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]

    return torch.tensor(keep, dtype=torch.long)
