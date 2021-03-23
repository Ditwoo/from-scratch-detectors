import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    with open(path, "r") as file:
        lines = file.read().split("\n")
        lines = [x for x in lines if x and not x.startswith("#")]
        lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
        module_defs = []
        for line in lines:
            if line.startswith("["):  # This marks the start of a new block
                module_defs.append({})
                module_defs[-1]["type"] = line[1:-1].rstrip()
                if module_defs[-1]["type"] == "convolutional":
                    module_defs[-1]["batch_normalize"] = 0
            else:
                key, value = line.split("=")
                value = value.strip()
                module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


def bbox_iou(box1, box2, order="xyxy"):
    """Returns the IoU of two bounding boxes

    Args: b
        box1 (torch.Tensor): tensor with shape [B, 4]
        box2 (torch.Tensor): tensor with shape [B, 4]
        order (str): boxes order, should be one of "xyxy"`, `"xywh"`.
            "xyxy" indicates that `box1` and `box2` has format (x1, y1, x2, y2) where
            (x1, y1) is a top left point and (x2, y2) is a bottom right point.
            "xywh" indicates that boxes has format (x, y, w, h) where
            (x, y) is a center of bbox and (w, h) - width and height of a box.
    """
    assert order in {"xyxy", "xywh"}

    if order == "xywh":
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_threshold):
    """[summary]

    Args:
        pred_boxes (torch.Tensor): [description]
        pred_cls (torch.Tensor): [description]
        target (torch.Tensor): [description]
        anchors (torch.Tensor): [description]
        ignore_threshold (float): IoU ignore threshold

    Returns:
        [type]: [description]
    """
    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = pred_boxes.new_full(size=(nB, nA, nG, nG), fill_value=0, dtype=torch.bool)
    noobj_mask = pred_boxes.new_full(size=(nB, nA, nG, nG), fill_value=1, dtype=torch.bool)
    class_mask = pred_boxes.new_full(size=(nB, nA, nG, nG), fill_value=0, dtype=torch.float)
    iou_scores = pred_boxes.new_full(size=(nB, nA, nG, nG), fill_value=0, dtype=torch.float)
    tx = pred_boxes.new_full(size=(nB, nA, nG, nG), fill_value=0, dtype=torch.float)
    ty = pred_boxes.new_full(size=(nB, nA, nG, nG), fill_value=0, dtype=torch.float)
    tw = pred_boxes.new_full(size=(nB, nA, nG, nG), fill_value=0, dtype=torch.float)
    th = pred_boxes.new_full(size=(nB, nA, nG, nG), fill_value=0, dtype=torch.float)
    tcls = pred_boxes.new_full(size=(nB, nA, nG, nG, nC), fill_value=0, dtype=torch.float)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]

    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)

    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()

    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set no_obj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_threshold, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()

    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1

    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, order="xywh")

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", nn.Sequential())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", nn.Sequential())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def forward(self, x, targets=None, img_dim=None):
        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, 5 + self.num_classes, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )  # [batch, anchors, grid, grid, 5 + num classes]

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.grid_size = grid_size
            self.stride = self.img_dim / self.grid_size

            # Calculate offsets for each grid
            self.grid_x = (
                torch.arange(self.grid_size, device=x.device)
                .repeat(self.grid_size, 1)
                .view([1, 1, self.grid_size, self.grid_size])
                .float()
            )
            self.grid_y = (
                torch.arange(self.grid_size, device=x.device)
                .repeat(self.grid_size, 1)
                .t()
                .view([1, 1, self.grid_size, self.grid_size])
                .float()
            )
            # fmt: off
            self.scaled_anchors = x.new_tensor([
                (a_w / self.stride, a_h / self.stride)
                for a_w, a_h in self.anchors
            ])
            # fmt: on
            self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
            self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = x.new_empty(prediction[..., :4].shape, dtype=torch.float)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, no_obj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_threshold=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = F.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = F.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = F.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = F.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = F.binary_cross_entropy(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_no_obj = F.binary_cross_entropy(pred_conf[no_obj_mask], tconf[no_obj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_no_obj
            loss_cls = F.binary_cross_entropy(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[no_obj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": total_loss.detach().cpu().item(),
                "x": loss_x.detach().cpu().item(),
                "y": loss_y.detach().cpu().item(),
                "w": loss_w.detach().cpu().item(),
                "h": loss_h.detach().cpu().item(),
                "conf": loss_conf.detach().cpu().item(),
                "cls": loss_cls.detach().cpu().item(),
                "cls_acc": cls_acc.detach().cpu().item(),
                "recall50": recall50.detach().cpu().item(),
                "recall75": recall75.detach().cpu().item(),
                "precision": precision.detach().cpu().item(),
                "conf_obj": conf_obj.detach().cpu().item(),
                "conf_no_obj": conf_noobj.detach().cpu().item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if isinstance(layer[0], YOLOLayer)]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        """
        Args:
            x (torch.Tensor): batch with data, expected shapes [B, 3, 416, 416]
            targets (torch.Tensor): targets, expected shapes [NUM_ANCHORS_IN_BATCH, 6],
                where [:, 0] is a vector with sample numbers, [:, 1] is a vector with
                bounding box class, [:, 2:] is a bounding box coordinates with
                structure (x, y, w, h) where (x,y) is a coordinates of bbox center and
                (w, h) - width and height.
                Default is `None`.

        Returns:
            If targets is `None` then returns only predictions
            (torch.Tensor with shapes [B, ANCHORS, 4 + NUM_CLASSES]),
            othervise returns loss and predictions.
        """
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)

        yolo_outputs = torch.cat(yolo_outputs, 1)

        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
        Args:
            path (str): path of the new weights file
            cutoff (int): save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


def yolov3_tiny(img_size=416, num_classes=80):
    curr_dir, _ = os.path.split(os.path.realpath(__file__))
    config = os.path.join(curr_dir, "configs", "yolov3-tiny.cfg")
    model = Darknet(config, img_size=img_size)
    return model
