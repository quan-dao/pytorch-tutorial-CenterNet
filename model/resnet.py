import torch
import torch.nn as nn
import torch.nn.functional as F

# from dcn_v2 import DCN
# assert torch.cuda.is_available(), "DCN doesn't work on cpu"


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
        self.conv1 = nn.Conv2d(inC, outC, kernel_size=3, stride=1 if not halve_size else 2,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outC)
        self.conv2 = nn.Conv2d(outC, outC, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outC)

        # components of skip connection
        self.conv_skip = None
        self.bn_skip = None
        if self.halve_size:
            self.conv_skip = nn.Conv2d(inC, outC, kernel_size=1, stride=2, padding=0, bias=False)
            self.bn_skip = nn.BatchNorm2d(outC)

    def forward(self, input):
        """Forward pass

        Args:
            input (torch.Tensor): shape (N, inC, H, W)
        Returns:
            torch.Tensor: shape (N, outC, H', W'), H' = H/2 if halve_size else H
        """
        # main pass
        out = self.conv1(input)  # shape (N, outC, H', W')
        out = F.relu(self.bn1(out))

        out = self.conv2(out)  # shape (N, outC, H', W')
        out = self.bn2(out)

        # skip connection
        if self.halve_size:
            residual = self.conv_skip(input)  # shape (N, outC, H/2, W/2)
            residual = self.bn_skip(residual)
        else:
            residual = input  # shape (N, inC, H, W)

        out = F.relu(out + residual)  # shape (N, outC, H', W')
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
        self.defconv = DCN(inC, outC, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(outC)
        self.convtrans = nn.ConvTranspose2d(outC, outC, kernel_size=4, stride=2, padding=1, output_padding=0,
                                            bias=False)
        self.bn2 = nn.BatchNorm2d(outC)

    def forward(self, input):
        """Forward pass

        Args:
            input (torch.Tensor): shape (N, inC, H, W)
        Return:
            torch.Tensor: upsampled tensor, shape (N, outC, 2*H, 2*W)
        """
        out = self.defconv(input)  # shape (N, outC, H, W)
        out = F.relu(self.bn1(out))
        out = self.convtrans(out)  # shape (N, outC, 2*H, 2*W)
        out = F.relu(self.bn2(out))
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # after passing through maxpool1, image size is reduced by 4

        # down-sampling path made of ResNet18
        self.conv2_1 = BasicLayer(64, 64)
        self.conv2_2 = BasicLayer(64, 64)

        self.conv3_1 = BasicLayer(64, 128, halve_size=True)
        self.conv3_2 = BasicLayer(128, 128)

        self.conv4_1 = BasicLayer(128, 256, halve_size=True)
        self.conv4_2 = BasicLayer(256, 256)

        self.conv5_1 = BasicLayer(256, 512, halve_size=True)
        self.conv5_2 = BasicLayer(512, 512)

        # up-sampling path
        self.up1 = UpSampleLayer(512, 256)
        self.up2 = UpSampleLayer(256, 128)
        self.up3 = UpSampleLayer(128, 64)

    # TODO: construct head

    def forward(self, input):
        """Forward pass

        Args:
            input (torch.Tensor): shape (N, 3, 384, 384)
        """
        out = self.conv1(input)  # (N, 64, 192, 192)
        out = F.relu(self.bn1(out))

        out = self.maxpool1(out)  # (N, 64, 96, 96)

        # down sampling path
        out = self.conv2_1(out)  # (N, 64, 96, 96)
        out = self.conv2_2(out)  # (N, 64, 96, 96)

        out = self.conv3_1(out)  # (N, 128, 48, 48)
        out = self.conv3_2(out)  # (N, 128, 48, 48)

        out = self.conv4_1(out)  # (N, 256, 24, 24)
        out = self.conv4_2(out)  # (N, 256, 24, 24)

        out = self.conv5_1(out)  # (N, 512, 12, 12)
        out = self.conv5_2(out)  # (N, 512, 12, 12)

        # up sampling path
        out = self.up1(out)  # (N, 256, 24, 24)
        out = self.up2(out)  # (N, 128, 48, 48)
        out = self.up3(out)  # (N, 64, 96, 96)

        return out


