import torch
import torch.nn as nn
import torch.nn.functional as F

from dcn_v2 import DCN
assert torch.cuda.is_available(), "DCN doesn't work on cpu"


class BasicLayer(nn.Module):
    """Basic residual layer used in ResNet18 & 34"""
    def __init__(self, inC, outC, halve_size=False):
        """BasicLayer's constructor

        Args:
            inC (int): number of channels of layer's input
            outC (int): number of channels of layer's output
            halve_size (bool): output has half the size of input or not. Default: False
        """
        super(BasicLayer, self).__init__()
        self.halve_size = halve_size
        # components of main pass
        self.conv1 = nn.Conv2d(inC, outC, kernel_size=0, stride=0,
                               padding=0, bias=False)  # TODO: declare value of kernel_size, stride, and padding
        self.bn1 = nn.BatchNorm2d(outC)
        self.conv2 = nn.Conv2d(outC, outC, kernel_size=0, stride=0,
                               padding=0, bias=False)  # TODO: declare value of kernel_size, stride, and padding
        self.bn2 = nn.BatchNorm2d(outC)

        # components of skip connection
        self.conv_skip = None
        self.bn_skip = None
        if self.halve_size:
            self.conv_skip = nn.Conv2d(inC, outC, kernel_size=0, stride=0,
                                       padding=0, bias=False)  # TODO: declare value of kernel_size, stride, and padding
            self.bn_skip = nn.BatchNorm2d(outC)

    def forward(self, input):
        """Forward pass TODO define the entire function
            TODO cont: remember to use BatchNorm2d declared as self.bnX after self.convX

        Args:
            input (torch.Tensor): shape (N, inC, H, W)
        Returns:
            torch.Tensor: shape (N, outC, H', W'), H' = H/2 if halve_size else H
        """
        # main pass
        out = self.conv1(...)  # shape (N, outC, H', W')
        out = F.relu(...)

        out = self.conv2(...)  # shape (N, outC, H', W')
        out = self.bn2(...)

        # skip connection
        if self.halve_size:
            residual = self.conv_skip(...)  # shape (N, outC, H/2, W/2)
            residual = self.bn_skip(...)
        else:
            residual = ...  # shape (N, inC, H, W)

        out = F.relu(...)  # shape (N, outC, H', W')
        return out


class UpSampleLayer(nn.Module):
    """A module made of a Deformable Convolution layer & a ConvTranspose2d for upsampling a feature map"""
    def __init__(self, inC, outC):
        """Constructor of UpSampleLayer

        Args:
            inC (int): number of input's channels
            outC (int): number of output's channels
        """
        super(UpSampleLayer, self).__init__()
        self.defconv = nn.Conv2d(inC, outC, kernel_size=0, stride=0,
                                 padding=0) # TODO: declare value of kernel_size, stride, and padding
        self.bn1 = nn.BatchNorm2d(outC)
        self.convtrans = nn.ConvTranspose2d(outC, outC, kernel_size=0, stride=0, padding=0, output_padding=0,
                                            bias=False)  # TODO: declare value of kernel_size, stride, and padding
        self.bn2 = nn.BatchNorm2d(outC)

    def forward(self, input):
        """Forward pass TODO: define the entire function
            TODO cont: remember to use BatchNorm2d declared as self.bnX after self.convX

        Args:
            input (torch.Tensor): shape (N, inC, H, W)
        Return:
            torch.Tensor: upsampled tensor, shape (N, outC, 2*H, 2*W)
        """
        out = torch.tensor([0])
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(..., ..., kernel_size=0, stride=0, padding=0,
                                bias=False)  # TODO: declare value of inC, outC, kernel_size, stride, and padding
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=0,
                                     stride=0, padding=0)  # TODO: declare value of kernel_size, stride, and padding
        # NOTE: after passing through maxpool1, image size is reduced by 4

        # down-sampling path made of ResNet18
        self.conv2_1 = BasicLayer(..., ..., ...)  # TODO: declare inC, outC, halve_size
        self.conv2_2 = BasicLayer(..., ..., ...)  # TODO: declare inC, outC, halve_size

        self.conv3_1 = BasicLayer(..., ..., ...)  # TODO: declare inC, outC, halve_size
        self.conv3_2 = BasicLayer(..., ..., ...)  # TODO: declare inC, outC, halve_size

        self.conv4_1 = BasicLayer(..., ..., ...)  # TODO: declare inC, outC, halve_size
        self.conv4_2 = BasicLayer(..., ..., ...)  # TODO: declare inC, outC, halve_size

        self.conv5_1 = BasicLayer(..., ..., ...)  # TODO: declare inC, outC, halve_size
        self.conv5_2 = BasicLayer(..., ..., ...)  # TODO: declare inC, outC, halve_size

        # up-sampling path
        # NOTE: compared to Figure.6 UpSampleLayer below are indexed from bottom to top
        # (meaning self.up3 is the last one)
        self.up1 = UpSampleLayer(..., ..., ...)  # TODO: declare inC, outC
        self.up2 = UpSampleLayer(..., ..., ...)  # TODO: declare inC, outC
        self.up3 = UpSampleLayer(..., ..., ...)  # TODO: declare inC, outC

    def forward(self, input):
        """Forward pass  TODO: define the entire function
            TODO cont: remember to use BatchNorm2d declared as self.bnX after self.convX

        Args:
            input (torch.Tensor): shape (N, 3, 384, 384)
        Returns:
            torch.Tensor: shape (N, 64, 96, 96)
        """
        out = torch.tensor([0])
        return out

