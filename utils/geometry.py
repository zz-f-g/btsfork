from typing import Literal, Tuple
import torch

EPS = 1e-3


def sample_coarse(rays: torch.Tensor, n_coarse: int, lindisp: bool):
    """
    Stratified sampling. Note this is different from original NeRF slightly.
    :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
    :return (B, Kc)
    """
    device = rays.device
    near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)

    step = 1.0 / n_coarse
    B = rays.shape[0]
    z_steps = torch.linspace(0, 1 - step, n_coarse, device=device)  # (Kc)
    z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # (B, Kc)
    z_steps += torch.rand_like(z_steps) * step
    if not lindisp:  # Use linear sampling in depth space
        return near * (1 - z_steps) + far * z_steps  # (B, Kc)
    else:  # Use linear sampling in disparity space
        return 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kc)


def make_points(
    rays: torch.Tensor,
    z_samp: torch.Tensor,
    super_batchsize: int,
    n_patch: int,
    y_patch: int,
    x_patch: int,
):
    B, K = z_samp.shape
    assert rays.shape == (B, 8)
    assert B == super_batchsize * n_patch * y_patch * x_patch

    rays = rays.view(super_batchsize * n_patch, y_patch * x_patch, 8)
    z_samp = z_samp.view(super_batchsize * n_patch, y_patch * x_patch, K).permute(
        0, 2, 1
    )
    points = rays[:, None, :, :3] + z_samp[..., None] * rays[:, None, :, 3:6]
    points = points.reshape(super_batchsize, B // super_batchsize * K, 3)

    return points


def project_3d(
    xyz: torch.Tensor,
    poses_w2c: torch.Tensor,
    Ks: torch.Tensor,
):
    assert xyz.shape[2] == 3
    n, n_pts, _ = xyz.shape
    nv = poses_w2c.shape[1]  # nv = 1 if encoder; nv = 4 if render
    assert poses_w2c.shape == (n, nv, 4, 4)
    assert Ks.shape == (n, nv, 3, 3)

    xyz = xyz[:, None, ...]
    xyz = torch.cat((xyz, torch.ones_like(xyz[..., :1])), dim=-1)  # [n, 1, n_pts, 4]
    xyz_encoder = poses_w2c[:, :, :3, :] @ xyz.permute(0, 1, 3, 2)  # [n, nv, 3, n_pts]
    distance = torch.norm(xyz_encoder, dim=-2)[
        ..., None
    ]  # [n, nv, n_pts, 1]
    uvz = (Ks @ xyz_encoder).permute(0, 1, 3, 2)  # [n, nv, n_pts, 3]
    uv = uvz[:, :, :, :2]
    z = uvz[:, :, :, 2:3]

    uv = uv / z.clamp_min(EPS)
    invalid = (
        (z <= EPS)
        | (uv[:, :, :, :1] < -1)
        | (uv[:, :, :, :1] > 1)
        | (uv[:, :, :, 1:2] < -1)
        | (uv[:, :, :, 1:2] > 1)
    )  # [n, nv, n_pts, 1]
    return uv, z, distance, invalid


def normalize_xyz(
    uv: torch.Tensor,
    z: torch.Tensor,
    distance: torch.Tensor,
    code_mode: Literal["z", "distance"],
    inv_z: bool,
    d_range: Tuple[float, float],
):
    assert uv.shape[1] == 1
    assert uv.shape[3] == 2
    n, _, n_pts, _ = uv.shape
    assert z.shape == (n, 1, n_pts, 1)
    assert distance.shape == (n, 1, n_pts, 1)
    assert code_mode in ("z", "distance")
    d_min, d_max = d_range

    if code_mode == "distance":
        z = distance

    if inv_z:
        z = (1 / z.clamp_min(EPS) - 1 / d_max) / (1 / d_min - 1 / d_max)
    else:
        z = (z - d_min) / (d_max - d_min)
    z = 2 * z - 1
    xyz = torch.cat((uv, z), dim=-1)
    return xyz
