import torch
import torch.nn as nn

from collections import OrderedDict


def conv_block(inC, midC, outC):
    """Construct a block of 2 conv2d layers. This block makes a branch in the head of an architecture
        in CenterNet family

    Args:
        inC (int): number of input channels
        midC (int): number of output channels of 1st conv2d layer
        outC (int): number of output channels of 2nd conv2d layer
    Returns:
        torch.nn.Module: a block of 2 conv2d layers
    """
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(inC, midC, kernel_size=0,
                            stride=0, padding=0, bias=True)),  # TODO: declare kernel_size, stride, padding
        ('relu', nn.ReLU()),
        ('conv2', nn.Conv2d(midC, outC, kernel_size=0,
                            stride=0, padding=0, bias=True))  # TODO: declare kernel_size, stride, padding
    ]))


class CenterNetHead(nn.Module):
    """Head of an architecture defined in CenterNet framework"""
    def __init__(self, inC, nClasses, midC):
        """Constructor of CenterNetHead

        Args:
            inC (int): number of channels of output of backbone
            nClasses (int): number of classes of objects in the dataset
            midC (int): number of output channels of 1st conv2d layer of a branch of head
        """
        super(CenterNetHead, self).__init__()
        self.heatmap = conv_block(inC, midC, ...)  # TODO: declare the outC
        self.wh = conv_block(inC, midC, ...)  # TODO: declare the outC
        self.offset = conv_block(inC, midC, ...)  # TODO: declare the outC

    def forward(self, input):
        """Forward pass

        Args:
            input (torch.Tensor): output of backone, shape (N, inC, H, W)
        """
        hm = self.heatmap(...)  # shape (N, nClasses, H, W) - TODO: declare input to self.heatmap()
        wh = self.wh(...)  # shape (N, 2, H, W) - TODO: declare input to self.wh()
        offset = self.offset(...)  # shape (N, 2, H, W) - TODO: declare input to self.offset()
        # TODO: concatenate hm, wh, offset to form the `out` tensor of shape (N, nClasses + 4, H, W)
        out = torch.cat(...)  # shape (N, nClasses + 4, H, W)
        return out


