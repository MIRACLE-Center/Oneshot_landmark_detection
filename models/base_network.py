import torch
import torchvision
import numpy as np
import torch.nn as nn
from torchvision.models import vgg19
# VGG = torchvision.models.vgg19
from torchvision.models.vgg import VGG, make_layers, cfgs, load_state_dict_from_url, model_urls


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

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    # model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    model = MyVgg(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


if __name__ == '__main__':
    net = vgg19(pretrained=True)
    x = torch.randn(10, 3, 256, 256)
    y, fea = net(x, get_features=True)
    # y, fea = net.features(x)
    print(y.shape)
    for f in fea:
        print(f.shape)
    import ipdb; ipdb.set_trace()