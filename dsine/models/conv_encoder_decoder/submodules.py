""" submodules
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import geffnet

INPUT_CHANNELS_DICT = {
    0: [1280, 112, 40, 24, 16],
    1: [1280, 112, 40, 24, 16],
    2: [1408, 120, 48, 24, 16],
    3: [1536, 136, 48, 32, 24],
    4: [1792, 160, 56, 32, 24],
    5: [2048, 176, 64, 40, 24],
    6: [2304, 200, 72, 40, 32],
    7: [2560, 224, 80, 48, 32]
}


class Encoder(nn.Module):
    def __init__(self, B=5, pretrained=True):
        """ e.g. B=5 will return EfficientNet-B5
        """
        super(Encoder, self).__init__()
        basemodel_name = 'tf_efficientnet_b%s_ap' % B
        basemodel = geffnet.create_model(basemodel_name, pretrained=pretrained)
        # Remove last layer
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        self.original_model = basemodel

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features, align_corners=True):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())
        self.align_corners = align_corners

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=self.align_corners)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class Conv2d_WS(nn.Conv2d):
    """ weight standardization
    """ 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class UpSampleGN(nn.Module):
    """ UpSample with GroupNorm
    """
    def __init__(self, skip_input, output_features, align_corners=True):
        super(UpSampleGN, self).__init__()
        self._net = nn.Sequential(Conv2d_WS(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(8, output_features),
                                  nn.LeakyReLU(),
                                  Conv2d_WS(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(8, output_features),
                                  nn.LeakyReLU())
        self.align_corners = align_corners

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=self.align_corners)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


def upsample_via_bilinear(out, up_mask, downsample_ratio):
    """ bilinear upsampling (up_mask is a dummy variable)
    """
    return F.interpolate(out, scale_factor=downsample_ratio, mode='bilinear', align_corners=False)


def upsample_via_mask(out, up_mask, downsample_ratio, padding='zero'):
    """ convex upsampling
    """
    # out: low-resolution output (B, o_dim, H, W)
    # up_mask: (B, 9*k*k, H, W)
    k = downsample_ratio

    B, C, H, W = out.shape
    up_mask = up_mask.view(B, 1, 9, k, k, H, W)
    up_mask = torch.softmax(up_mask, dim=2)             # (B, 1, 9, k, k, H, W)

    if padding == 'zero':
        # with zero padding
        up_out = F.unfold(out, [3, 3], padding=1)       # (B, 2, H, W) -> (B, 2 X 3*3, H*W)
    elif padding == 'replicate':
        # with replicate padding
        out = F.pad(out, pad=(1,1,1,1), mode='replicate')
        up_out = F.unfold(out, [3, 3], padding=0)           # (B, C, H, W) -> (B, C X 3*3, H*W)
    else:
        raise Exception('invalid padding for convex upsampling')

    up_out = up_out.view(B, C, 9, 1, 1, H, W)           # (B, C, 9, 1, 1, H, W)

    up_out = torch.sum(up_mask * up_out, dim=2)         # (B, C, k, k, H, W)
    up_out = up_out.permute(0, 1, 4, 2, 5, 3)           # (B, C, H, k, W, k)
    return up_out.reshape(B, C, k*H, k*W)               # (B, C, kH, kW)


def dummy_activation(out):
    return out


def get_prediction_head(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Conv2d(input_dim, hidden_dim, 3, padding=1), 
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_dim, hidden_dim, 1), 
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_dim, output_dim, 1),
    )


class Decoder(nn.Module):
    def __init__(self, num_classes=2, 
                 B=5, NF=2048, BN=True,
                 down=2, learned_upsampling=True,
                 activation_fn=dummy_activation):
        super(Decoder, self).__init__()
        input_channels = INPUT_CHANNELS_DICT[B]

        # use BN or GN
        UpSample = UpSampleBN if BN else UpSampleGN

        features = NF
        self.conv2 = nn.Conv2d(input_channels[0], features, kernel_size=1, stride=1, padding=0)
        self.up1 = UpSample(skip_input=features // 1 + input_channels[1], output_features=features // 2)
        self.up2 = UpSample(skip_input=features // 2 + input_channels[2], output_features=features // 4)

        if down == 8:
            i_dim = features // 4
        elif down == 4:
            self.up3 = UpSample(skip_input=features // 4 + input_channels[3], output_features=features // 8)
            i_dim = features // 8
        elif down == 2:
            self.up3 = UpSample(skip_input=features // 4 + input_channels[3], output_features=features // 8)
            self.up4 = UpSample(skip_input=features // 8 + input_channels[4], output_features=features // 16)
            i_dim = features // 16
        else:
            raise Exception('invalid downsampling ratio')

        self.downsample_ratio = down
        self.output_dim = num_classes

        h_dim = 128
        self.pred_head = get_prediction_head(i_dim, h_dim, num_classes)

        if learned_upsampling:
            h_dim = 128
            self.mask_head = get_prediction_head(i_dim, h_dim, 9*self.downsample_ratio*self.downsample_ratio)
            self.upsample_fn = upsample_via_mask
        else:
            self.mask_head = lambda a: None
            self.upsample_fn = upsample_via_bilinear

        self.activation_fn = activation_fn


    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)

        if self.downsample_ratio == 8:
            x_feat = self.up2(x_d1, x_block2)
        elif self.downsample_ratio == 4:
            x_d2 = self.up2(x_d1, x_block2)
            x_feat = self.up3(x_d2, x_block1)
        elif self.downsample_ratio == 2:
            x_d2 = self.up2(x_d1, x_block2)
            x_d3 = self.up3(x_d2, x_block1)
            x_feat = self.up4(x_d3, x_block0)

        out = self.activation_fn(self.pred_head(x_feat))
        mask = self.mask_head(x_feat)
        up_out = self.upsample_fn(out, mask, self.downsample_ratio)
        return up_out

