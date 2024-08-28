"""
Main model implementation
"""

# import ipdb
from torch.func import vmap, grad
import torch
import torch.autograd.profiler as profiler
import torch.nn.functional as F
from torch import nn

from models.common.backbones.backbone_util import make_backbone
from models.common.model.code import PositionalEncoding
from models.common.model.mlp_util import make_mlp

EPS = 1e-3


class BTSNet(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.d_min = conf.get("z_near")
        self.d_max = conf.get("z_far")

        self.learn_empty = conf.get("learn_empty", True)
        self.empty_empty = conf.get("empty_empty", False)
        self.inv_z = conf.get("inv_z", True)

        self.color_interpolation = conf.get("color_interpolation", "bilinear")
        self.code_mode = conf.get("code_mode", "z")
        if self.code_mode not in ["z", "distance"]:
            raise NotImplementedError(f"Unknown mode for positional encoding: {self.code_mode}")

        self.encoder = make_backbone(conf["encoder"])
        self.code_xyz = PositionalEncoding.from_conf(conf["code"], d_in=3)

        self.flip_augmentation = conf.get("flip_augmentation", False)

        self.return_sample_depth = conf.get("return_sample_depth", False)

        self.sample_color = conf.get("sample_color", True)

        d_in = self.encoder.latent_size + self.code_xyz.d_out
        d_out = 1 if self.sample_color else 4

        self._d_in = d_in
        self._d_out = d_out

        self.mlp_coarse = make_mlp(conf["mlp_coarse"], d_in, d_out=d_out)
        self.mlp_fine = make_mlp(conf["mlp_fine"], d_in, d_out=d_out, allow_empty=True)

        if self.learn_empty:
            self.empty_feature = nn.Parameter(torch.randn((self.encoder.latent_size,), requires_grad=True))

        self._scale = 0

    def set_scale(self, scale):
        self._scale = scale

    def get_scale(self):
        return self._scale

    def compute_grid_transforms(self, *args, **kwargs):
        pass

    def encode(self, images, Ks, poses_c2w, ids_encoder=None, ids_render=None, images_alt=None, combine_ids=None):
        poses_w2c = torch.inverse(poses_c2w)

        if ids_encoder is None:
            images_encoder = images
            Ks_encoder = Ks
            poses_w2c_encoder = poses_w2c
            ids_encoder = list(range(len(images)))
        else:
            # 4.1
            images_encoder = images[:, ids_encoder] # Use image at [-1, 1] as input for ENC
            Ks_encoder = Ks[:, ids_encoder]
            poses_w2c_encoder = poses_w2c[:, ids_encoder]

        if images_alt is not None:
            images = images_alt
        else:
            images = images * .5 + .5

        if ids_render is None:
            images_render = images
            Ks_render = Ks
            poses_w2c_render = poses_w2c
            ids_render = list(range(len(images)))
        else:
            # 4.2
            images_render = images[:, ids_render]
            Ks_render = Ks[:, ids_render]
            poses_w2c_render = poses_w2c[:, ids_render]

        if combine_ids is not None:
            combine_ids = list(list(group) for group in combine_ids)
            get_combined = set(sum(combine_ids, []))
            for i in range(images.shape[1]):
                if i not in get_combined:
                    combine_ids.append((i,))
            remap_encoder = {v: i for i, v in enumerate(ids_encoder)}
            remap_render = {v: i for i, v in enumerate(ids_render)}
            comb_encoder = [[remap_encoder[i] for i in group if i in ids_encoder] for group in combine_ids]
            comb_render = [[remap_render[i] for i in group if i in ids_render] for group in combine_ids]
            comb_encoder = [group for group in comb_encoder if len(group) > 0]
            comb_render = [group for group in comb_render if len(group) > 0]
        else:
            comb_encoder = None
            comb_render = None

        n, nv, c, h, w = images_encoder.shape
        c_l = self.encoder.latent_size

        if self.flip_augmentation and self.training:
            do_flip = (torch.rand(1) > .5).item()
        else:
            do_flip = False

        if do_flip:
            images_encoder = torch.flip(images_encoder, dims=(-1, ))

        # 4.3
        image_latents_ms = self.encoder(images_encoder.view(n * nv, c, h, w))

        if do_flip:
            image_latents_ms = [torch.flip(il, dims=(-1, )) for il in image_latents_ms]

        _, _, h_, w_ = image_latents_ms[0].shape
        # 4.4
        image_latents_ms = [F.interpolate(image_latents, (h_, w_)).view(n, nv, c_l, h_, w_) for image_latents in image_latents_ms]

        # 4.5
        self.grid_f_features = image_latents_ms
        self.grid_f_Ks = Ks_encoder
        self.grid_f_poses_w2c = poses_w2c_encoder
        self.grid_f_combine = comb_encoder

        self.grid_c_imgs = images_render
        self.grid_c_Ks = Ks_render
        self.grid_c_poses_w2c = poses_w2c_render
        self.grid_c_combine = comb_render

    def sample_features(self, xyz, use_single_featuremap=True):
        n, n_pts, _ = xyz.shape
        n, nv, c, h, w = self.grid_f_features[self._scale].shape

        # if use_single_featuremap:
        #     nv = 1

        xyz = xyz.unsqueeze(1)  # (n, 1, pts, 3)
        ones = torch.ones_like(xyz[..., :1])
        xyz = torch.cat((xyz, ones), dim=-1) # (n, 1, pts, 4)
        xyz_projected = ((self.grid_f_poses_w2c[:, :nv, :3, :]) @ xyz.permute(0, 1, 3, 2)) # 612411 (n, nv, 3, pts)
        distance = torch.norm(xyz_projected, dim=-2).unsqueeze(-1) # (n, nv, pts) ?
        xyz_projected = (self.grid_f_Ks[:, :nv] @ xyz_projected).permute(0, 1, 3, 2) # (n, nv, pts, 3)
        xy = xyz_projected[:, :, :, [0, 1]] # (n, nv, pts, 2)
        z = xyz_projected[:, :, :, 2:3] # (n, nv, pts, 1)

        xy = xy / z.clamp_min(EPS) # 612412
        invalid = (z <= EPS) | (xy[:, :, :, :1] < -1) | (xy[:, :, :, :1] > 1) | (xy[:, :, :, 1:2] < -1) | (xy[:, :, :, 1:2] > 1) # 612413 (n, nv, pts, 1)

        if self.code_mode == "z":
            # Get z into [-1, 1] range
            if self.inv_z:
                z = (1 / z.clamp_min(EPS) - 1 / self.d_max) / (1 / self.d_min - 1 / self.d_max)
            else:
                z = (z - self.d_min) / (self.d_max - self.d_min)
            z = 2 * z - 1
            xyz_projected = torch.cat((xy, z), dim=-1) # (n, nv, pts, 3)
        elif self.code_mode == "distance":
            if self.inv_z:
                distance = (1 / distance.clamp_min(EPS) - 1 / self.d_max) / (1 / self.d_min - 1 / self.d_max)
            else:
                distance = (distance - self.d_min) / (self.d_max - self.d_min)
            distance = 2 * distance - 1
            xyz_projected = torch.cat((xy, distance), dim=-1)
        xyz_code = self.code_xyz(xyz_projected.view(n * nv * n_pts, -1)).view(n, nv, n_pts, -1) # 612414

        feature_map = self.grid_f_features[self._scale][:, :nv] # 612415 (n, nv, c, h, w)
        # These samples are from different scales
        if self.learn_empty:
            empty_feature_expanded = self.empty_feature.view(1, 1, 1, c).expand(n, nv, n_pts, c)

        sampled_features = F.grid_sample(feature_map.view(n * nv, c, h, w), xy.view(n * nv, 1, -1, 2), mode="bilinear", padding_mode="border", align_corners=False).view(n, nv, c, n_pts).permute(0, 1, 3, 2) # 612416 (n, nv, n_pts, c)

        if self.learn_empty:
            sampled_features[invalid.expand(-1, -1, -1, c)] = empty_feature_expanded[invalid.expand(-1, -1, -1, c)]

        sampled_features = torch.cat((sampled_features, xyz_code), dim=-1) # 612417

        # If there are multiple frames with predictions, reduce them.
        # TODO: Technically, this implementations should be improved if we use multiple frames.
        # The reduction should only happen after we perform the unprojection.

        if self.grid_f_combine is not None:
            invalid_groups = []
            sampled_features_groups = []

            for group in self.grid_f_combine:
                if len(group) == 1:
                    invalid_groups.append(invalid[:, group])
                    sampled_features_groups.append(sampled_features[:, group])

                invalid_to_combine = invalid[:, group]
                features_to_combine = sampled_features[:, group]

                indices = torch.min(invalid_to_combine, dim=1, keepdim=True)[1]
                invalid_picked = torch.gather(invalid_to_combine, dim=1, index=indices)
                features_picked = torch.gather(features_to_combine, dim=1, index=indices.expand(-1, -1, -1, features_to_combine.shape[-1]))

                invalid_groups.append(invalid_picked)
                sampled_features_groups.append(features_picked)

            invalid = torch.cat(invalid_groups, dim=1)
            sampled_features = torch.cat(sampled_features_groups, dim=1)

        if use_single_featuremap:
            sampled_features = sampled_features.mean(dim=1)
            invalid = torch.any(invalid, dim=1)

        return sampled_features, invalid

    def sample_colors(self, xyz):
        n, n_pts, _ = xyz.shape
        n, nv, c, h, w = self.grid_c_imgs.shape
        xyz = xyz.unsqueeze(1)                      # (n, 1, pts, 3)
        ones = torch.ones_like(xyz[..., :1])
        xyz = torch.cat((xyz, ones), dim=-1)
        xyz_projected = ((self.grid_c_poses_w2c[:, :, :3, :]) @ xyz.permute(0, 1, 3, 2))
        distance = torch.norm(xyz_projected, dim=-2).unsqueeze(-1)
        xyz_projected = (self.grid_c_Ks @ xyz_projected).permute(0, 1, 3, 2)
        xy = xyz_projected[:, :, :, [0, 1]]
        z = xyz_projected[:, :, :, 2:3]

        # This scales the x-axis into the right range.
        xy = xy / z.clamp_min(EPS)
        invalid = (z <= EPS) | (xy[:, :, :, :1] < -1) | (xy[:, :, :, :1] > 1) | (xy[:, :, :, 1:2] < -1) | (xy[:, :, :, 1:2] > 1)

        sampled_colors = F.grid_sample(self.grid_c_imgs.view(n * nv, c, h, w), xy.view(n * nv, 1, -1, 2), mode=self.color_interpolation, padding_mode="border", align_corners=False).view(n, nv, c, n_pts).permute(0, 1, 3, 2)
        assert not torch.any(torch.isnan(sampled_colors))

        if self.grid_c_combine is not None:
            invalid_groups = []
            sampled_colors_groups = []

            for group in self.grid_c_combine:
                if len(group) == 1:
                    invalid_groups.append(invalid[:, group])
                    sampled_colors_groups.append(sampled_colors[:, group])
                    continue

                invalid_to_combine = invalid[:, group]
                colors_to_combine = sampled_colors[:, group]

                indices = torch.min(invalid_to_combine, dim=1, keepdim=True)[1]
                invalid_picked = torch.gather(invalid_to_combine, dim=1, index=indices)
                colors_picked = torch.gather(colors_to_combine, dim=1, index=indices.expand(-1, -1, -1, colors_to_combine.shape[-1]))

                invalid_groups.append(invalid_picked)
                sampled_colors_groups.append(colors_picked)

            invalid = torch.cat(invalid_groups, dim=1)
            sampled_colors = torch.cat(sampled_colors_groups, dim=1)

        if self.return_sample_depth:
            distance = distance.view(n, nv, n_pts, 1)
            sampled_colors = torch.cat((sampled_colors, distance), dim=-1)

        return sampled_colors, invalid

    def forward(self, xyz, coarse=True, viewdirs=None, far=False, only_density=False):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (B, 3)
        B is batch of points (in rays)
        :return (B, 4) r g b sigma
        """

        with profiler.record_function("model_inference"):
            n, n_pts, _ = xyz.shape
            nv = self.grid_c_imgs.shape[1]

            if self.grid_c_combine is not None:
                nv = len(self.grid_c_combine)

            # Sampled features all has shape: scales [n, n_pts, c + xyz_code]
            # ipdb.set_trace() # 61241
            sampled_features, invalid_features = self.sample_features(xyz, use_single_featuremap=not only_density)                  # invalid features (n, n_pts, 1)
            sampled_features = sampled_features.reshape(n * n_pts, -1)

            mlp_input = sampled_features.view(n, n_pts, -1)

            # Camera frustum culling stuff, currently disabled
            combine_index = None
            dim_size = None

            # Run main NeRF network
            if coarse or self.mlp_fine is None:
                # ipdb.set_trace() # 61242
                mlp_output = self.mlp_coarse(
                    mlp_input, # (n, n_pts, -1)
                    combine_inner_dims=(n_pts,),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )
            else:
                mlp_output = self.mlp_fine(
                    mlp_input,
                    combine_inner_dims=(n_pts,),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )

            # (n, pts, c) -> (n, n_pts, c)
            mlp_output = mlp_output.reshape(n, n_pts, self._d_out)

            if self.sample_color:
                # ipdb.set_trace() # 61243
                sigma = mlp_output[..., :1]
                sigma = F.softplus(sigma)
                # ipdb.set_trace() # 61244
                rgb, invalid_colors = self.sample_colors(xyz)                               # (n, nv, pts, 3)
            else:
                sigma = mlp_output[..., :1]
                sigma = F.relu(sigma)
                rgb = mlp_output[..., 1:4].reshape(n, 1, n_pts, 3)
                rgb = F.sigmoid(rgb)
                invalid_colors = invalid_features.unsqueeze(-2)
                nv = 1

            if self.empty_empty:
                sigma[invalid_features[..., 0]] = 0
            # TODO: Think about this!
            # Since we don't train the colors directly, lets use softplus instead of relu

            if not only_density:
                _, _, _, c = rgb.shape
                rgb = rgb.permute(0, 2, 1, 3).reshape(n, n_pts, nv * c)         # (n, pts, nv * 3)
                invalid_colors = invalid_colors.permute(0, 2, 1, 3).reshape(n, n_pts, nv)

                invalid = invalid_colors | invalid_features                 # Invalid features gets broadcasted to (n, n_pts, nv)
                invalid = invalid.to(rgb.dtype)
            else:
                rgb = torch.zeros((n, n_pts, nv * 3), device=sigma.device)
                invalid = invalid_features.to(sigma.dtype)
        # ipdb.set_trace()  # sigma grad
        # grad_sigma = torch.autograd.grad(sigma[0, 0], xyz)
        g = get_sigma_grad(
            xyz,
            self.grid_f_features[self._scale][:, :nv],
            self.grid_f_poses_w2c,
            self.grid_f_Ks,
            self.d_min,
            self.d_max,
            n_freqs=6,
            freq_factor=1.5,
            network=self.mlp_coarse,
        )
        # get_normal_each_pt = lambda xyz_: grad(get_sigma_each_pt)(
        #     xyz_,
        #     feature_map[0],
        #     self.grid_f_poses_w2c[0],
        #     self.grid_f_Ks[0],
        #     self.d_min,
        #     self.d_max,
        #     n_freqs=6,
        #     freq_factor=1.5,
        #     network=self.mlp_coarse,
        # )
        # grad_sigma = torch.stack([get_normal_each_pt(p) for p in xyz[0]], dim=0)
        # ipdb.set_trace()  # 61245
        return rgb, invalid, sigma, g


def positional_encode(x: torch.Tensor, n_freqs: int, freq_factor: float):
    from math import pi

    phases = torch.zeros(2 * n_freqs, device=x.device)
    phases[1::2] = pi * 0.5
    phases = phases.view(1, -1, 1)
    freqs = torch.repeat_interleave(
        freq_factor * 2.0 ** torch.arange(0, n_freqs, device=x.device), repeats=2
    ).view(1, -1, 1)
    emb = x.unsqueeze(1).repeat(1, 2 * n_freqs, 1)
    emb = torch.sin(torch.addcmul(phases, emb, freqs))
    return torch.cat((x, emb.view(x.shape[0], -1)), dim=-1)


def get_sigma_grad(
    xyz: torch.Tensor,
    feature: torch.Tensor,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
    d_min: int,
    d_max: int,
    n_freqs: int,
    freq_factor: int,
    network: nn.Module,
):
    n, n_pts, _ = xyz.shape
    n_, nv, c, h, w = feature.shape
    assert n == n_
    xyz_ = xyz.unsqueeze(1)
    ones = torch.ones_like(xyz_[..., :1])
    xyz_ = torch.cat((xyz_, ones), dim=-1)
    xyz_projected = (
        K[:, :nv] @ (pose_w2c[:, :nv, :3, :] @ (xyz_.permute(0, 1, 3, 2)))
    ).permute(0, 1, 3, 2)
    xy = xyz_projected[..., :2] / xyz_projected[..., 2:].clamp_min(EPS)
    sampled_features = (
        F.grid_sample(
            feature.view(n * nv, c, h, w),
            xy.view(n * nv, 1, -1, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        .view(n, nv, c, n_pts)
        .permute(0, 3, 1, 2)
    )

    def get_sigma_each_pt(
        xyz: torch.Tensor,
        feature_pt: torch.Tensor,
        pose_w2c: torch.Tensor,
        K: torch.Tensor,
    ):
        assert xyz.shape == (3,)
        assert len(feature_pt.shape) == 2
        assert len(pose_w2c.shape) == 3
        assert len(K.shape) == 3
        assert pose_w2c.shape[1:] == (4, 4)
        assert K.shape[1:] == (3, 3)
        nv, c = feature_pt.shape
        xyz = xyz.unsqueeze(0)  # (1, 3)
        xyz = torch.cat((xyz, torch.ones_like(xyz[..., :1])), dim=-1)  # (1, 4)
        xyz_projected = pose_w2c[:nv, :3, :] @ xyz.unsqueeze(-1)
        xyz_projected = (K[:nv] @ xyz_projected).squeeze(-1)  # (nv, 3)
        xy = xyz_projected[:, [0, 1]]  # (nv, 2)
        z = xyz_projected[:, 2:3]  # (nv, 1)
        xy = xy / z.clamp_min(1e-3)
        z = (1 / z.clamp_min(1e-3) - 1 / d_max) / (1 / d_min - 1 / d_max)
        z = 2 * z - 1
        xyz_projected = torch.cat((xy, z), dim=-1)  # (nv, 3)
        xyz_code = positional_encode(xyz_projected, n_freqs, freq_factor)  # (nv, 39)
        sampled_features = torch.cat((feature_pt, xyz_code), dim=-1)  # (nv, c')
        mlp_input = sampled_features.mean(dim=0).view(1, -1)  # use_single_featuremap
        mlp_output = network(mlp_input)
        sigma = F.softplus(mlp_output[..., 0].squeeze())
        return sigma

    def get_sigma_grad_each_batch(xyz, feature_pt, pose_w2c, K):
        return vmap(grad(get_sigma_each_pt), in_dims=(0, 0, None, None))(xyz, feature_pt, pose_w2c, K)

    return vmap(get_sigma_grad_each_batch)(xyz, sampled_features, pose_w2c, K)
