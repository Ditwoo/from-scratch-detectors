import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet


class ResNet(nn.Module):
    def __init__(self, backbone="resnet50", backbone_path=None):
        """

        Args:
            backbone (str): resnet backbone to use.
                Expected one of ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
                Default is "resnet50".
            backbone_path (str): path to pretrained backbone model.
                If ``None`` then will be used torchvision pretrained model.
                Default is None.
        """
        super().__init__()
        if backbone == "resnet18":
            backbone = resnet.resnet18(pretrained=not backbone_path)
            self.out_channels = [256, 512, 512, 256, 256, 128]
        elif backbone == "resnet34":
            backbone = resnet.resnet34(pretrained=not backbone_path)
            self.out_channels = [256, 512, 512, 256, 256, 256]
        elif backbone == "resnet50":
            backbone = resnet.resnet50(pretrained=not backbone_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == "resnet101":
            backbone = resnet.resnet101(pretrained=not backbone_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == "resnet152":
            backbone = resnet.resnet152(pretrained=not backbone_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        else:
            raise ValueError(f"Unknown ResNet backbone - '{backbone}'!")

        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SSD300(nn.Module):
    def __init__(self, backbone="resnet50", n_classes=81):
        """
        Source:
            https://github.com/NVIDIA/DeepLearningExamples/blob/70fcb70ff4bc49cc723195b35cfa8d4ce94a7f76/PyTorch/Detection/SSD/src/model.py

        Args:
            backbone (str): model backbone to use
            n_classes (int): number of classes to predict
        """
        super().__init__()

        self.feature_extractor = ResNet(backbone)

        self.label_num = n_classes  # number of COCO classes
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.loc = []
        self.conf = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(oc, nd * self.label_num, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self._init_weights()

    def _build_additional_features(self, input_size):
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(
            zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])
        ):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            # ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.label_num, -1)))
            ret.append((l(s).view(s.size(0), -1, 4), c(s).view(s.size(0), -1, self.label_num)))

        locs, confs = list(zip(*ret))
        # locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        locs, confs = torch.cat(locs, 1).contiguous(), torch.cat(confs, 1).contiguous()
        return locs, confs

    def forward(self, x):
        x = self.feature_extractor(x)

        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        return locs, confs


class Loss(nn.Module):
    def __init__(self, num_classes, ignore_class=0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_class = ignore_class

    def _hard_negative_mining(self, cls_loss, pos):
        """Return negative indices that is 3x the number as positive indices.

        Args:
            cls_loss: (torch.Tensor) cross entropy loss between `cls_preds` and `cls_targets`.
                Expected shape [B, M] where B - batch, M - anchors.
            pos: (torch.Tensor) positive class mask.
                Expected shape [B, M] where B - batch, M - anchors.

        Return:
            (torch.Tensor) negative indices, sized [N,#anchors].
        """
        cls_loss = cls_loss * (pos.float() - 1)

        _, idx = cls_loss.sort(1)  # sort by negative losses
        _, rank = idx.sort(1)  # [B, M]

        num_neg = 3 * pos.sum(1)  # [B,]
        neg = rank < num_neg[:, None]  # [B, M]
        return neg

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        """Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        loss:
            loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(cls_preds, cls_targets).

        Args:
            loc_preds: (torch.Tensor) predicted locations.
                Expected shapes - [B, M, 4] where B - batch, M - anchors.
            loc_targets: (torch.Tensor) encoded target locations.
                Expected shapes - [B, M, 4] where B - batch, M - anchors.
            cls_preds: (torch.Tensor) predicted class confidences.
                Expected shapes - [B, M, CLASS] where B - batch, M - anchors, CLASS - number of classes.
            cls_targets: (torch.LongTensor) encoded target labels.
                Expected shapes - [B, M] where B - batch, M - anchors.

        Returns:
            regression loss and classification loss
        """
        pos = cls_targets > 0  # [B, M]
        batch_size = pos.size(0)
        num_pos = pos.sum().item()

        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [B, M, 4]
        loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask], reduction="sum")

        cls_loss = F.cross_entropy(
            cls_preds.view(-1, self.num_classes), cls_targets.view(-1), reduction="none"
        )  # [B * M,]
        cls_loss = cls_loss.view(batch_size, -1)
        cls_loss[cls_targets == self.ignore_class] = 0  # set ignored loss to 0
        neg = self._hard_negative_mining(cls_loss, pos)  # [B, M]
        cls_loss = cls_loss[pos | neg].sum()

        loc_loss = loc_loss / num_pos
        cls_loss = cls_loss / num_pos

        return loc_loss, cls_loss