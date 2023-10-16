import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

import torchvision
from torchvision.models._utils import handle_legacy_interface, _ovewrite_named_param
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.transforms._presets import ImageClassification

from torchvision.models.vgg import cfgs, make_layers, VGG

_COMMON_META = {
    "min_size": (32, 32),
    "categories": _IMAGENET_CATEGORIES,
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
    "_docs": """These weights were trained from scratch by using a simplified training recipe.""",
}

class MyVgg(VGG):
    def __init__(self, *args, **kwargs):
        print("Load MyVgg")
        super(MyVgg, self).__init__(*args, **kwargs)

    def forward(self, input, get_features=False):
        features_list = []
        for module in self.features._modules.values():
            input = module(input)
            if get_features:
                if 'MaxPool2d' in module.__repr__():
                    features_list.append(input)
        if get_features:
            return input, features_list
        else:
            return input

    def get_features(self, input):
        return self.forward(input, get_features=True)

class VGG19_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 143667240,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 72.376,
                    "acc@5": 90.876,
                }
            },
            "_ops": 19.632,
            "_file_size": 548.051,
        },
    )
    DEFAULT = IMAGENET1K_V1


def _vgg(cfg: str, batch_norm: bool, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> VGG:
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = MyVgg(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    return model

@handle_legacy_interface(weights=("pretrained", VGG19_Weights.IMAGENET1K_V1))
def vgg19(*, weights: Optional[VGG19_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-19 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG19_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG19_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG19_Weights
        :members:
    """
    weights = VGG19_Weights.verify(weights)

    return _vgg("E", False, weights, progress, **kwargs)

class Rela_Pos_Regression_Net(nn.Module):
    def __init__(self, inc=128, num_dim=2):
        super(Rela_Pos_Regression_Net, self).__init__()
        self.fc1 = nn.Linear(inc, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_dim)
        # self.fc1 = nn.Linear(inc, inc)
        # self.fc2 = nn.Linear(inc, inc)
        # self.fc3 = nn.Linear(inc, num_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class UNet_Pretrained(nn.Module):
    def __init__(self, n_channels, non_local=False, length_embedding=64):
        super(UNet_Pretrained, self).__init__()
        self.n_channels = n_channels
        bilinear = True
        length_embedding = length_embedding

        self.vgg =  vgg19(pretrained=True)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 128, bilinear)
        self.up5 = Up(128, 128, bilinear)
        self.trans_5 = nn.Conv2d(512, length_embedding, kernel_size=1, padding=0)
        self.trans_4 = nn.Conv2d(256, length_embedding, kernel_size=1, padding=0)
        self.trans_3 = nn.Conv2d(128, length_embedding, kernel_size=1, padding=0)
        self.trans_2 = nn.Conv2d(64, length_embedding, kernel_size=1, padding=0)
        self.trans_1 = nn.Conv2d(128, length_embedding, kernel_size=1, padding=0)
        self.trans_0 = nn.Conv2d(128, length_embedding, kernel_size=1, padding=0)

        if non_local:
            self.non_local_5 = RFB_modified(512, 512)
            self.non_local_4 = RFB_modified(256, 256)
            self.non_local_3 = RFB_modified(128, 128)
        self.non_local = non_local

        self.RPR_head = Rela_Pos_Regression_Net(inc=128)
    

    def forward(self, x, mlp=False, get_feature=False):
        _, features = self.vgg.features(x, get_features=True)
        # # For torchvsion of later version 
        # features = self.vgg.features(x)

        x = self.up1(features[4], features[3])
        if self.non_local:
            fea_5 = self.trans_5(self.non_local_5(features[4]))
            fea_4 = self.trans_4(self.non_local_4(x))
        else:
            fea_5 = self.trans_5(features[4])
            fea_4 = self.trans_4(x)
        rela_feature = x

        x = self.up2(x, features[2])
        if self.non_local:
            fea_3 = self.trans_3(self.non_local_3(x))
        else:
            fea_3 = self.trans_3(x)
        
        
        x = self.up3(x, features[1])
        fea_2 = self.trans_2(x)

        x = self.up4(x, features[0])
        fea_1 = self.trans_1(x)

        fea_0 = self.trans_0(self.up5(x))

        if get_feature: return  rela_feature

        return [fea_5, fea_4, fea_3, fea_2, fea_1, fea_0]
    
    def fowward_rela(self, x):
        return self.RPR_head(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)

class OutConv_Sigmoid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_Sigmoid, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, bn_layer=True):
        super(NonLocalBlock, self).__init__()


        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

if __name__ == '__main__':
    test = UNet_Pretrained(3, 57)
    wtf = torch.zeros([1, 3, 224, 224], dtype=torch.float)
    wtf = test(wtf)
    import ipdb; ipdb.set_trace()

