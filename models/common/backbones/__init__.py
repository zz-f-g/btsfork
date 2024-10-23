# from .backbone_util import *

from omegaconf import OmegaConf
from torch import nn
from mmseg.models.backbones import (
    ResNet,
    SwinTransformer,
    BiSeNetV1,
    BiSeNetV2,
    MobileNetV2,
    MobileNetV3,
    ResNetV1c,
)
from mmseg.models.decode_heads import (
    FCNHead,
    PSPHead,
    ASPPHead,
    FPNHead,
    DepthwiseSeparableASPPHead,
    UPerHead,
)
from mmpretrain.models import ConvNeXt
# from mmseg.models import backbones
# from mmseg.models import decode_heads

from models.common.backbones.monodepth2 import Monodepth2
from models.common.backbones.efficientnet import EffiBackbone


class Network(nn.Module):
    """
    Encoder-decoder-based network for image feature extraction.
    Input:
    - images: Tensor of shape (N, 3, H, W)

    Output:
    - features: Tensor of shape (N, C, H, W)
    """

    def __init__(self, config):
        super(Network, self).__init__()
        config = OmegaConf.to_object(config)
        self.latent_size = config.get("d_out", 64)
        self.enc_kwargs = config.get("encoder")
        self.dec_kwargs = config.get("decoder")
        self.dec_kwargs.update({"num_classes": self.latent_size})

        self.enc_name = self.enc_kwargs.pop("type")
        self.dec_name = self.dec_kwargs.pop("type")
        # self.encoder = getattr(backbones, self.enc_name)(**self.enc_kwargs)
        # self.decoder = getattr(decode_heads, self.dec_name)(**self.dec_kwargs)
        self.encoder = globals()[self.enc_name](**self.enc_kwargs)
        self.decoder = globals()[self.dec_name](**self.dec_kwargs)

        print("encoder kwargs: ", self.enc_kwargs)
        print("encoder name: ", self.enc_name)
        print("decoder name: ", self.dec_name)

    def forward(self, rgb):
        out = self.encoder(rgb)
        out = self.decoder(out)
        return out


def make_backbone(conf):
    if conf.get("type") == "monodepth2":
        net = Monodepth2.from_conf(conf)
    elif conf.get("type") == "efficientnet":
        net = EffiBackbone.from_conf(conf)
    elif conf.get("mono2") == False:
        net = Network(conf)
    else:
        ValueError("Invalid backbone name")

    return net
