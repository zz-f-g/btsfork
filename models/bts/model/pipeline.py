from typing import List
import torch
from torch import Tensor
import torch.nn.functional as F

from utils.timer import profiler
from utils import geometry
from utils import render
from models.common.backbones.efficientnet import EffiBackbone
from .e2edecoder import decoder, e2edecoder
from models.bts.model.models_bts import BTSNet


def make_pipeline(
    backbone: EffiBackbone,
    net: BTSNet,
    n_z: int,
    lindisp: bool,
    use_flip_aug: bool,
    is_training: bool,
    patches_each_batch: int,
    noise_std: float,
):
    def pipeline(
        images: Tensor,
        images_ip: Tensor,
        all_rays: Tensor,
        poses: Tensor,
        projs: Tensor,
        ids_encoder: List[int],
        ids_render: List[int],
        n_patch: int,
        y_patch: int,
        x_patch: int,
    ):
        n, _, c, h, w = images.shape
        with profiler.record_function("trainer_prepare-points"):
            all_rays = all_rays.reshape(-1, 8)
            z_coarse = geometry.sample_coarse(
                all_rays,
                n_z,
                lindisp,
            )
            points = geometry.make_points(
                all_rays,
                z_coarse,
                n,
                n_patch,
                y_patch,
                x_patch,
            )
            _, n_pts, _ = points.shape

        with profiler.record_function("trainer_sample-colors"):
            uv_render, z_render, distance_render, invalid_render = geometry.project_3d(
                points,
                torch.inverse(poses)[:, ids_render],
                projs[:, ids_render],
            )
            nv_render = len(ids_render)
            invalid_render = invalid_render.permute(0, 2, 1, 3).reshape(
                n, n_pts, nv_render
            )
            sampled_colors = (
                F.grid_sample(
                    images_ip[:, ids_render].view(n * nv_render, c, h, w),
                    uv_render.view(n * nv_render, 1, -1, 2),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
                .view(n, nv_render, c, n_pts)
                .permute(0, 1, 3, 2)
            )

        with profiler.record_function("trainer_encoder"):
            nv_encoder = len(ids_encoder)
            images_encoder = images[:, ids_encoder]
            if use_flip_aug and is_training:
                do_flip = (torch.rand(1) > 0.5).item()
            else:
                do_flip = False
            if do_flip:
                images_encoder = torch.flip(images_encoder, dims=(-1,))
            layer_features = []
            encoder_feature = backbone.encoder[0](
                images_encoder.view(n * nv_encoder, c, h, w)
            )
            for i in range(1, len(backbone.encoder)):
                encoder_feature = backbone.encoder[i](encoder_feature)
                if i in backbone.out_layers:
                    layer_features.append(encoder_feature)
            # encoder_out = backbone.decoder(layer_features)
            # image_latents_ms = [
            #     encoder_out[("disp", i)] for i in range(backbone.scales)
            # ]
            # if do_flip:
            #     image_latents_ms = [
            #         torch.flip(il, dims=(-1,)) for il in image_latents_ms
            #     ]
            # _, _, h_, w_ = image_latents_ms[0].shape
            # c_l = backbone.latent_size
            # image_latents_ms = [
            #     F.interpolate(image_latents, (h_, w_)).view(n, nv_encoder, c_l, h_, w_)
            #     for image_latents in image_latents_ms
            # ]
            # net.grid_f_features = image_latents_ms  # for evaluation
            if do_flip:
                layer_features = [torch.flip(il, dims=(-1, )) for il in layer_features]

        with profiler.record_function("trainer_decoder"):
            patches_split_size = (patches_each_batch - 1) // n + 1
            split_points = torch.split(points.view(n, n_patch, n_z, y_patch, x_patch, 3), patches_split_size, dim=1)
            poses_encoder, projs_encoder = poses[:, ids_encoder], projs[:, ids_encoder]
            sigma = []
            invalid_encoder = []
            for pnts in split_points:
                sigma_, invalid_encoder_ = e2edecoder(
                    net,
                    backbone.decoder,
                    pnts.view(n, -1, 3),
                    poses_encoder,
                    projs_encoder,
                    layer_features,
                    y_patch,
                    x_patch,
                )
                sigma.append(sigma_)
                invalid_encoder.append(invalid_encoder_)
            sigma = torch.cat(sigma, dim=1)
            invalid_encoder = torch.cat(invalid_encoder, dim=1)

        with profiler.record_function("trainer_nerf-render"):
            if is_training and noise_std > 0.0:
                sigma = sigma + torch.rand_like(sigma) * noise_std
            weights, rgb_final, depth_final, alpha, z_samp, rgbs = render.nerf_render(
                z_coarse,
                (
                    sampled_colors.view(n, nv_render, n_patch, n_z, y_patch, x_patch, c)
                    .permute(0, 2, 4, 5, 3, 1, 6)
                    .reshape(n * n_patch * y_patch * x_patch, n_z, nv_render * c)
                ),
                (
                    sigma.view(n, n_patch, n_z, y_patch, x_patch)
                    .permute(0, 1, 3, 4, 2)
                    .reshape(n * n_patch * y_patch * x_patch, n_z)
                ),
                nv=nv_render,
            )
            invalid = (
                (invalid_encoder | invalid_render)
                .to(z_coarse.dtype)
                .reshape(*z_coarse.shape, nv_render)
            )
            invalid = (
                (invalid_encoder | invalid_render)
                .to(z_coarse.dtype)
                .reshape(*z_coarse.shape, nv_render)
            )
        return (weights, rgb_final, depth_final, alpha, invalid, z_samp, rgbs)

    return pipeline


if __name__ == "__main__":
    ppl = make_pipeline()

"""
with profiler.record_function("trainer_prepare-points"):
    superbatch_size = all_rays.shape[0]
    all_rays = all_rays.reshape(-1, 8)
    z_coarse = geometry.sample_coarse(
        all_rays,
        self.renderer.renderer.n_coarse,
        self.renderer.renderer.lindisp,
    )
    points = geometry.make_points(all_rays, z_coarse, superbatch_size)
    _, n_pts, _ = points.shape

with profiler.record_function("trainer_sample-colors"):
    uv_render, z_render, distance_render, invalid_render = geometry.project_3d(
        points,
        torch.inverse(poses)[:, ids_render],
        projs[:, ids_render],
    )
    nv_render = len(ids_render)
    invalid_render = invalid_render.permute(0, 2, 1, 3).reshape(n, n_pts, nv_render)
    sampled_colors = (
        F.grid_sample(
            images_ip[:, ids_render].view(n * nv_render, c, h, w),
            uv_render.view(n * nv_render, 1, -1, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        .view(n, nv_render, c, n_pts)
        .permute(0, 1, 3, 2)
    )

with profiler.record_function("trainer_encoder"):
    nv_encoder = len(ids_encoder)
    images_encoder = images[:, ids_encoder]
    if self.renderer.net.flip_augmentation and self.training:
        do_flip = (torch.rand(1) > .5).item()
    else:
        do_flip = False
    if do_flip:
        images_encoder = torch.flip(images_encoder, dims=(-1, ))
    layer_features = []
    encoder_feature = self.backbone.encoder[0](images_encoder.view(n * nv_encoder, c, h, w))
    for i in range(1, len(self.backbone.encoder)):
        encoder_feature = self.backbone.encoder[i](encoder_feature)
        if i in self.backbone.out_layers:
            layer_features.append(encoder_feature)
    encoder_out = self.backbone.decoder(layer_features)
    image_latents_ms = [encoder_out[("disp", i)] for i in range(self.backbone.scales)]
    if do_flip:
        image_latents_ms = [torch.flip(il, dims=(-1, )) for il in image_latents_ms]
    _, _, h_, w_ = image_latents_ms[0].shape
    c_l = self.backbone.latent_size
    image_latents_ms = [F.interpolate(image_latents, (h_, w_)).view(n, nv_encoder, c_l, h_, w_) for image_latents in image_latents_ms]
    # if do_flip:
    #     layer_features = [torch.flip(il, dims=(-1, )) for il in layer_features]

with profiler.record_function("trainer_decoder"):
    eval_batch_size = (self.renderer.renderer.eval_batch_size - 1) // superbatch_size + 1
    eval_batch_dim = 1
    split_points = torch.split(points, eval_batch_size, dim=eval_batch_dim)
    poses_encoder, projs_encoder = poses[:, ids_encoder], projs[:, ids_encoder]
    sigma = []
    invalid_encoder = []
    for pnts in split_points:
        sigma_, invalid_encoder_ = decoder(
            self.renderer.net,
            pnts,
            poses_encoder,
            projs_encoder,
            image_latents_ms,
        ) # TODO:
        sigma.append(sigma_)
        invalid_encoder.append(invalid_encoder_)
    sigma = torch.cat(sigma, dim=eval_batch_dim)
    invalid_encoder = torch.cat(invalid_encoder, dim=eval_batch_dim)

with profiler.record_function("trainer_nerf-render"):
    if self.training and self.renderer.renderer.noise_std > 0.0:
        sigma = sigma + torch.rand_like(sigma) * self.renderer.net.noise_std
    weights, rgb_final, depth_final, alpha, z_samp, rgbs = (
        render.nerf_render(
            z_coarse,
            sampled_colors.permute(0, 2, 1, 3).reshape(*z_coarse.shape, nv_render * c),
            sigma.reshape(*z_coarse.shape),
            nv=nv_render,
        )
    )
    invalid = (invalid_encoder | invalid_render).to(z_coarse.dtype).reshape(*z_coarse.shape, nv_render)
"""
