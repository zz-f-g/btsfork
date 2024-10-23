from pathlib import Path
from typing import List, Optional
from utils.timer import profiler
from models.common.model.layers import Conv3x3, ConvBlock

from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class MonodepthDecoder(nn.Module):
    def __init__(
        self,
        num_ch_enc: List[int],
        num_ch_dec: Optional[List[int]],
        d_out: int,
        scales: int,
        use_skips: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = num_ch_dec if num_ch_dec else [128, 128, 256, 256, 512]
        self.d_out = d_out
        self.num_ch_dec = [max(d_out, chns) for chns in self.num_ch_dec]
        self.scales = scales
        self.use_skips = use_skips

        # decoder
        self.convs = OrderedDict()
        for i in range(self.scales, -1, -1):
            # upconv_0
            num_ch_in = (
                self.num_ch_enc[-1] if i == self.scales else self.num_ch_dec[i + 1]
            )
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in range(self.scales):
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.d_out)

        self.decoder_keys = {k: i for i, k in enumerate(self.convs.keys())}
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features: List[torch.Tensor]):
        self.outputs = {}

        x = input_features[-1]
        for i in range(self.scales, -1, -1):
            x = self.decoder[self.decoder_keys[("upconv", i, 0)]](x)

            x = F.interpolate(x, scale_factor=(2, 2), mode="nearest")

            if self.use_skips and i > 0:
                feats = input_features[i - 1]
                x = torch.cat([x, feats], 1)

            x = self.decoder[self.decoder_keys[("upconv", i, 1)]](x)

            self.outputs[("features", i)] = x

            if i in range(self.scales):
                self.outputs[("disp", i)] = self.decoder[
                    self.decoder_keys[("dispconv", i)]
                ](x)
        return self.outputs


class EffiBackbone(nn.Module):
    def __init__(
        self,
        cp_location: Optional[Path] = None,
        freeze: bool = False,
        num_ch_dec: Optional[List[int]] = None,
        d_out: int = 64,
        out_layers: List[int] = [1, 2, 3, 5, 8],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.out_layers = out_layers
        self.encoder = models.efficientnet_b0(
            weights=models.efficientnet.EfficientNet_B0_Weights.IMAGENET1K_V1
        ).features
        self.layer_channels = [
            32, 16,  # [96, 320]
            24,  # [48, 160]
            40,  # [24, 80]
            80, 112,  # [12, 40]
            192, 320, 1280,  # [6, 20]
        ]
        num_ch_enc = [self.layer_channels[layer_idx] for layer_idx in self.out_layers]
        self.scales = len(out_layers) - 1
        self.decoder = MonodepthDecoder(num_ch_enc, num_ch_dec, d_out, self.scales)

        self.latent_size = d_out

        if cp_location is not None:
            cp = torch.load(cp_location)
            self.load_state_dict(cp["model"])

        if freeze:
            for p in self.parameters(True):
                p.requires_grad = False

    def forward(self, images: torch.Tensor):
        with profiler.record_function("backbone_forward"):
            layer_features = []
            with profiler.record_function("encoder_forward"):
                for i in range(len(self.encoder)):
                    images = self.encoder[i](images)
                    if i in self.out_layers:
                        layer_features.append(images)
            with profiler.record_function("decoder_forward"):
                out = self.decoder(layer_features)
            x = [out[("disp", i)] for i in range(self.scales)]
        return x

    @classmethod
    def from_conf(cls, conf):
        return cls(
            cp_location=conf.get("cp_location", None),
            freeze=conf.get("freeze", False),
            num_ch_dec=conf.get("num_ch_dec", None),
            d_out=conf.get("d_out", 128),
            out_layers=conf.get("out_layers", [1, 2, 3, 5, 8]),
        )
