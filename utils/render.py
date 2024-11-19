import torch


def nerf_render(
    z_samp: torch.Tensor,
    rgbs: torch.Tensor,
    sigmas: torch.Tensor,
    nv: int
):
    B, K =  z_samp.shape
    assert rgbs.shape == (B, K, nv * 3)
    assert sigmas.shape == (B, K)
    # if self.training and self.noise_std > 0.0: # TODO:
    #     sigmas = sigmas + torch.randn_like(sigmas) * self.noise_std

    deltas = z_samp[:, 1:] - z_samp[:, :-1]  # (B, K-1)
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # infty (B, 1)
    # delta_inf = rays[:, -1:] - z_samp[:, -1:]
    deltas = torch.cat([deltas, delta_inf], -1)  # (B, K)

    alphas = 1 - torch.exp(
        -deltas.abs() * torch.relu(sigmas)
    )  # (B, K) (delta should be positive anyways)

    alphas[:, -1] = 1

    deltas = None
    sigmas = None
    alphas_shifted = torch.cat(
        [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
    )  # (B, K+1) = [1, a1, a2, ...]
    T = torch.cumprod(alphas_shifted, -1)  # (B)
    weights = alphas * T[:, :-1]  # (B, K)
    # alphas = None
    alphas_shifted = None

    rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (B, 3)
    depth_final = torch.sum(weights * z_samp, -1)  # (B)

    return weights, rgb_final, depth_final, alphas, z_samp, rgbs
