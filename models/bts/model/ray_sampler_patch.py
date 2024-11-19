import torch
from torch import Tensor
from omegaconf import ListConfig
from typing import List, Tuple

from models.common.util import util
from .ray_sampler import RaySampler


class ImagePatchRaySampler(RaySampler):
    def __init__(
        self,
        z_near: float,
        z_far: float,
        patch_size: int | Tuple[int, int] | List[int] | ListConfig,
        height: int,
        width: int,
        channels: int = 3,
        norm_dir: bool = True,
    ) -> None:
        self.z_near = z_near
        self.z_far = z_far
        if isinstance(patch_size, int):
            self.patch_size_x, self.patch_size_y = patch_size, patch_size
        elif (
            isinstance(patch_size, tuple)
            or isinstance(patch_size, list)
            or isinstance(patch_size, ListConfig)
        ):
            self.patch_size_y = patch_size[0]
            self.patch_size_x = patch_size[1]
        else:
            raise ValueError(f"Invalid format for patch size")
        self.height = height
        self.width = width
        self.channels = channels
        self.norm_dir = norm_dir
        assert (height % self.patch_size_y) == 0
        assert (width % self.patch_size_x) == 0

    def sample(self, images: Tensor, poses: Tensor, projs: Tensor):
        n, v, _, _ = poses.shape

        self._patch_count = (
            v * self.height * self.width // (self.patch_size_x * self.patch_size_y)
        )
        all_rgb_gt = []
        all_rays = []

        focals = projs[:, :, [0, 1], [0, 1]].view(n * v, 2)
        centers = projs[:, :, [0, 1], [2, 2]].view(n * v, 2)

        all_rays = (
            util.gen_rays(
                poses.view(n * v, 4, 4),
                self.width,
                self.height,
                focal=focals,
                c=centers,
                z_near=self.z_near,
                z_far=self.z_far,
                norm_dir=self.norm_dir,
            )
            .view(
                n * v,
                self.height // self.patch_size_y,
                self.patch_size_y,
                self.width // self.patch_size_x,
                self.patch_size_x,
                8,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(n, self._patch_count * self.patch_size_y * self.patch_size_x, 8)
        )
        all_rgb_gt = (
            images.view(
                n * v,
                3,
                self.height // self.patch_size_y,
                self.patch_size_y,
                self.width // self.patch_size_x,
                self.patch_size_x,
            )
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(n, self._patch_count * self.patch_size_y * self.patch_size_x, 3)
        )

        return all_rays, all_rgb_gt

    def reconstruct(self, render_dict, channels=None):
        coarse = render_dict["coarse"]
        fine = render_dict["fine"]

        if channels is None:
            channels = self.channels

        c_rgb = coarse["rgb"]  # n, n_pts, v * 3
        c_weights = coarse["weights"]
        c_depth = coarse["depth"]
        c_invalid = coarse["invalid"]

        f_rgb = fine["rgb"]  # n, n_pts, v * 3
        f_weights = fine["weights"]
        f_depth = fine["depth"]
        f_invalid = fine["invalid"]

        n, n_pts, v_c = c_rgb.shape
        v_in = n_pts // (self.height * self.width)
        v_render = v_c // channels
        c_n_smps = c_weights.shape[-1]
        f_n_smps = f_weights.shape[-1]
        # (This can be a different v from the sample method)

        coarse["rgb"] = (
            c_rgb.view(
                n,
                v_in,
                self.height // self.patch_size_y,
                self.width // self.patch_size_x,
                self.patch_size_y,
                self.patch_size_x,
                v_render,
                channels,
            )
            .permute(0, 1, 2, 4, 3, 5, 6, 7)
            .reshape(n, v_in, self.height, self.width, v_render, channels)
        )
        coarse["weights"] = (
            c_weights.view(
                n,
                v_in,
                self.height // self.patch_size_y,
                self.width // self.patch_size_x,
                c_n_smps,
                self.patch_size_y,
                self.patch_size_x,
            )
            .permute(0, 1, 2, 5, 3, 6, 4)
            .reshape(n, v_in, self.height, self.width, c_n_smps)
        )
        coarse["depth"] = (
            c_depth.view(
                n,
                v_in,
                self.height // self.patch_size_y,
                self.width // self.patch_size_x,
                self.patch_size_y,
                self.patch_size_x,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(n, v_in, self.height, self.width)
        )
        coarse["invalid"] = (
            c_invalid.view(
                n,
                v_in,
                self.height // self.patch_size_y,
                self.width // self.patch_size_x,
                c_n_smps,
                self.patch_size_y,
                self.patch_size_x,
                v_render,
            )
            .permute(0, 1, 2, 5, 3, 6, 4, 7)
            .reshape(n, v_in, self.height, self.width, c_n_smps, v_render)
        )

        fine["rgb"] = (
            f_rgb.view(
                n,
                v_in,
                self.height // self.patch_size_y,
                self.width // self.patch_size_x,
                self.patch_size_y,
                self.patch_size_x,
                v_render,
                channels,
            )
            .permute(0, 1, 2, 4, 3, 5, 6, 7)
            .reshape(n, v_in, self.height, self.width, v_render, channels)
        )
        fine["weights"] = (
            f_weights.view(
                n,
                v_in,
                self.height // self.patch_size_y,
                self.width // self.patch_size_x,
                f_n_smps,
                self.patch_size_y,
                self.patch_size_x,
            )
            .permute(0, 1, 2, 5, 3, 6, 4)
            .reshape(n, v_in, self.height, self.width, f_n_smps)
        )
        fine["depth"] = (
            f_depth.view(
                n,
                v_in,
                self.height // self.patch_size_y,
                self.width // self.patch_size_x,
                self.patch_size_y,
                self.patch_size_x,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(n, v_in, self.height, self.width)
        )
        fine["invalid"] = (
            f_invalid.view(
                n,
                v_in,
                self.height // self.patch_size_y,
                self.width // self.patch_size_x,
                f_n_smps,
                self.patch_size_y,
                self.patch_size_x,
                v_render,
            )
            .permute(0, 1, 2, 5, 3, 6, 4, 7)
            .reshape(n, v_in, self.height, self.width, f_n_smps, v_render)
        )

        if "alphas" in coarse:
            c_alphas = coarse["alphas"]
            f_alphas = fine["alphas"]
            coarse["alphas"] = (
                c_alphas.view(
                    n,
                    v_in,
                    self.height // self.patch_size_y,
                    self.width // self.patch_size_x,
                    c_n_smps,
                    self.patch_size_y,
                    self.patch_size_x,
                )
                .permute(0, 1, 2, 5, 3, 6, 4)
                .reshape(n, v_in, self.height, self.width, c_n_smps)
            )
            fine["alphas"] = (
                f_alphas.view(
                    n,
                    v_in,
                    self.height // self.patch_size_y,
                    self.width // self.patch_size_x,
                    f_n_smps,
                    self.patch_size_y,
                    self.patch_size_x,
                )
                .permute(0, 1, 2, 5, 3, 6, 4)
                .reshape(n, v_in, self.height, self.width, f_n_smps)
            )

        if "z_samps" in coarse:
            c_z_samps = coarse["z_samps"]
            f_z_samps = fine["z_samps"]
            coarse["z_samps"] = (
                c_z_samps.view(
                    n,
                    v_in,
                    self.height // self.patch_size_y,
                    self.width // self.patch_size_x,
                    c_n_smps,
                    self.patch_size_y,
                    self.patch_size_x,
                )
                .permute(0, 1, 2, 5, 3, 6, 4)
                .reshape(n, v_in, self.height, self.width, c_n_smps)
            )
            fine["z_samps"] = (
                f_z_samps.view(
                    n,
                    v_in,
                    self.height // self.patch_size_y,
                    self.width // self.patch_size_x,
                    f_n_smps,
                    self.patch_size_y,
                    self.patch_size_x,
                )
                .permute(0, 1, 2, 5, 3, 6, 4)
                .reshape(n, v_in, self.height, self.width, f_n_smps)
            )

        if "rgb_samps" in coarse:
            c_rgb_samps = coarse["rgb_samps"]
            f_rgb_samps = fine["rgb_samps"]
            coarse["rgb_samps"] = (
                c_rgb_samps.view(
                    n,
                    v_in,
                    self.height // self.patch_size_y,
                    self.width // self.patch_size_x,
                    c_n_smps,
                    self.patch_size_y,
                    self.patch_size_x,
                    v_render,
                    channels,
                )
                .permute(0, 1, 2, 5, 3, 6, 4, 7, 8)
                .reshape(n, v_in, self.height, self.width, c_n_smps, v_render, channels)
            )
            fine["rgb_samps"] = (
                f_rgb_samps.view(
                    n,
                    v_in,
                    self.height // self.patch_size_y,
                    self.width // self.patch_size_x,
                    f_n_smps,
                    self.patch_size_y,
                    self.patch_size_x,
                    v_render,
                    channels,
                )
                .permute(0, 1, 2, 5, 3, 6, 4, 7, 8)
                .reshape(n, v_in, self.height, self.width, f_n_smps, v_render, channels)
            )

        render_dict["coarse"] = coarse
        render_dict["fine"] = fine

        if "rgb_gt" in render_dict:
            rgb_gt = render_dict["rgb_gt"]
            render_dict["rgb_gt"] = (
                rgb_gt.view(
                    n,
                    v_in,
                    self.height // self.patch_size_y,
                    self.width // self.patch_size_x,
                    self.patch_size_y,
                    self.patch_size_x,
                    channels,
                )
                .permute(0, 1, 2, 4, 3, 5, 6)
                .reshape(n, v_in, self.height, self.width, channels)
            )

        return render_dict
