import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torchvision.utils import List
from utils import geometry
from utils.timer import profiler
from models.bts.model.models_bts import BTSNet
from models.common.backbones.efficientnet import MonodepthDecoder


def downsample(input_tensor: Tensor, scale: int, y_patch: int, x_patch: int):
    n, nv, n_pts, _ = input_tensor.shape
    return (
        F.interpolate(
            input_tensor.reshape(-1, y_patch, x_patch, 2).permute(0, 3, 1, 2),
            scale_factor=0.5**scale,
            mode="bilinear",
            align_corners=False,
        )
        .permute(0, 2, 3, 1)
        .reshape(n * nv, n_pts // 2 ** (2 * scale), 2)
    )


class EffiMLP(nn.Module):
    def __init__(self, feature_chnls: List[int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.feature_chnls = feature_chnls
        lins = []
        for i in range(1, len(feature_chnls)):
            lin = nn.Linear(feature_chnls[i - 1], feature_chnls[i])
            nn.init.kaiming_normal_(lin.weight, a=0, mode="fan_in")
            nn.init.constant_(lin.bias, 0.0)
            lins.append(lin)
        self.lins = nn.ModuleList(lins)
        self.activation = nn.ReLU(inplace=True)
        self.lastlin = nn.Linear(feature_chnls[-1], 64)
        nn.init.kaiming_normal_(self.lastlin.weight, a=0, mode="fan_in")
        nn.init.constant_(self.lastlin.bias, 0.0)

    def forward(self, layer_features):
        layer_features_ = layer_features[::-1]
        assert self.feature_chnls == [t.shape[-1] for t in layer_features_]
        for i in range(len(self.lins)):
            if i == 0:
                x = self.lins[i](layer_features_[i])
            else:
                x = self.lins[i](x + layer_features_[i])
            x = self.activation(x)
        x = self.lastlin(x + layer_features_[-1])
        return x


def decoder(
    net: BTSNet,
    points: Tensor,
    poses: Tensor,
    projs: Tensor,
    feature_map: List[Tensor],
):
    n, n_pts, _ = points.shape
    nv = poses.shape[1]
    assert poses.shape == (n, nv, 4, 4)
    assert projs.shape == (n, nv, 3, 3)
    assert feature_map[0].shape[:2] == (n, nv)
    _, _, cf, hf, wf = feature_map[0].shape

    with profiler.record_function("decoder_positional-encoding"):
        uv_encoder, z_encoder, distance_encoder, invalid_encoder = geometry.project_3d(
            points,
            torch.inverse(poses),
            projs,
        )
        if net.code_mode == "z":
            z_tocode = z_encoder
        elif net.code_mode == "distance":
            z_tocode = distance_encoder
        else:
            raise ValueError(net.code_mode)
        # Get z into [-1, 1] range
        if net.inv_z:
            z_tocode = (1 / z_tocode.clamp_min(geometry.EPS) - 1 / net.d_max) / (
                1 / net.d_min - 1 / net.d_max
            )
        else:
            z_tocode = (z_tocode - net.d_min) / (net.d_max - net.d_min)
        z_tocode = 2 * z_tocode - 1
        uvz_normalized = torch.cat((uv_encoder, z_tocode), dim=-1)  # (n, nv, n_pts, 1)

    with profiler.record_function("decoder_prepare-mlpinput"):
        # feature_map = self.renderer.net.grid_f_features[0][:, :nv]
        sampled_features = (
            F.grid_sample(
                feature_map[0].view(n * nv, cf, hf, wf),
                uv_encoder.view(n * nv, 1, n_pts, 2),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            .view(n, nv, cf, n_pts)
            .permute(0, 1, 3, 2)
        )
        uvz_code = net.code_xyz(uvz_normalized.view(n * nv * n_pts, 3)).view(
            n, nv, n_pts, -1
        )
        mlp_input = torch.cat(
            (sampled_features, uvz_code),
            dim=-1,
        ).mean(
            dim=1
        )  # (n, n_pts, 103)

    with profiler.record_function("decoder_mlp"):
        mlp_output = net.mlp_coarse(
            mlp_input,
            combine_inner_dims=(n_pts,),
            combine_index=None,
            dim_size=None,
        ).view(n, n_pts)
        sigma = F.softplus(mlp_output)
    return sigma, torch.any(invalid_encoder, dim=1)  # (n, n_pts), # (n, n_pts, 1)


def e2edecoder(
    net: BTSNet,
    decoder: MonodepthDecoder,
    points: Tensor,
    poses: Tensor,
    projs: Tensor,
    layer_features: List[Tensor],
    y_patch: int,
    x_patch: int,
):
    n, n_pts, _ = points.shape
    nv = poses.shape[1]
    assert poses.shape == (n, nv, 4, 4)
    assert projs.shape == (n, nv, 3, 3)
    assert layer_features[0].shape[0] == n * nv

    with profiler.record_function("decoder_positional-encoding"):
        uv_encoder, z_encoder, distance_encoder, invalid_encoder = geometry.project_3d(
            points,
            torch.inverse(poses),
            projs,
        )
        if net.code_mode == "z":
            z_tocode = z_encoder
        elif net.code_mode == "distance":
            z_tocode = distance_encoder
        else:
            raise ValueError(net.code_mode)
        # Get z into [-1, 1] range
        if net.inv_z:
            z_tocode = (1 / z_tocode.clamp_min(geometry.EPS) - 1 / net.d_max) / (
                1 / net.d_min - 1 / net.d_max
            )
        else:
            z_tocode = (z_tocode - net.d_min) / (net.d_max - net.d_min)
        z_tocode = 2 * z_tocode - 1
        uvz_normalized = torch.cat((uv_encoder, z_tocode), dim=-1)  # (n, nv, n_pts, 3)

    with profiler.record_function("decoder_prepare-mlpinput"):
        # feature_map = self.renderer.net.grid_f_features[0][:, :nv]
        sampled_features = []
        for i in range(1, 1 + len(layer_features)):
            sampled_features.append(
                F.grid_sample(
                    layer_features[i-1],
                    downsample(uv_encoder, i, y_patch, x_patch)[:, None, ...],
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
                .view(
                    n * nv,
                    -1,
                    n_pts // (y_patch * x_patch),
                    y_patch // 2**i,
                    x_patch // 2**i,
                )
                .permute(0, 2, 1, 3, 4)
                .reshape(
                    n * nv * n_pts // (y_patch * x_patch),
                    -1,
                    y_patch // 2**i,
                    x_patch // 2**i,
                )
            )
        uvz_code = net.code_xyz(uvz_normalized.view(n * nv * n_pts, 3)).view(
            n, nv, n_pts, -1
        )

    with profiler.record_function("decoder_net-forward"):
        feature = (
            decoder(sampled_features)[("disp", 0)]
            .view(n, nv, n_pts // (y_patch * x_patch), -1, y_patch, x_patch)
            .permute(0, 1, 2, 4, 5, 3)
            .reshape(n, nv, n_pts, -1)
        )
        mlp_input = torch.cat(
            (feature, uvz_code),
            dim=-1,
        ).mean(dim=1)
        mlp_output = net.mlp_coarse(
            mlp_input,
            combine_inner_dims=(n_pts,),
            combine_index=None,
            dim_size=None,
        ).view(n, n_pts)
        sigma = F.softplus(mlp_output)
    return sigma, torch.any(invalid_encoder, dim=1)  # (n, n_pts), # (n, n_pts, 1)
