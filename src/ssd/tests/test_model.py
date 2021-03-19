import torch
from ssd.model import SSD300


def test_ssd300_forward():
    for backbone in (18, 34, 50, 101):
        l, c = SSD300(f"resnet{backbone}", 123).eval().forward(torch.randn(1, 3, 300, 300))
        assert l.shape == (1, 8732, 4)
        assert c.shape == (1, 8732, 123)
