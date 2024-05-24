import torch
from torch import nn

class Synthesizer:
    def __init__(self) -> None:
        self.synthesize_strategy = self.volume_render_channel
        self.speedOfLight = 299792458

    def synthesize(
            self,     
            raw: torch.Tensor,
            z_vals: torch.Tensor,
            fc: torch.Tensor,
            ray_batches = None
        ):
        return self.synthesize_strategy(raw, z_vals, fc, ray_batches)


    ####################### Helper functions #########################
    def cumprod_exclusive(
        self,
        tensor: torch.Tensor,
        dim
    ) -> torch.Tensor:
        r"""
        (Courtesy of https://github.com/krrish94/nerf-pytorch)

        Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

        Args:
        tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
        is to be computed.
        Returns:
        cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
        tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
        """
        # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
        cumprod = torch.cumprod(tensor, dim)
        # "Roll" the elements along dimension 'dim' by 1 element.
        cumprod = torch.roll(cumprod, 1, dim)
        # Replace the first element by "0".
        cumprod[:, 0] = torch.zeros(tensor.shape[:-1])

        return cumprod
    
    def free_space_path_loss(self, dist, freq):
        return 20*torch.log10(dist) + 20*torch.log10(freq)+20*torch.log10(torch.tensor(4*torch.pi/self.speedOfLight))
    
    #####################################################################
    def volume_render_channel(
        self,
        raw: torch.Tensor,
        z_vals: torch.Tensor,
        fc: torch.Tensor,
        ray_batches
    ):
        # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, 1e-10 * torch.ones_like(dists[..., :1])], dim=-1)
        alpha = 1.0-torch.exp(-nn.functional.relu(raw[..., -1]) * dists)  

        weights = alpha * self.cumprod_exclusive((1. - alpha) + 1e-10, dim = 1)
        depth = (weights * z_vals).sum(dim = -1) # [n_rays]
        phs_shift = torch.exp(-1j*(2*torch.pi*fc*1e9/self.speedOfLight)*dists)
        # Compute weight for each sample along each ray. [n_rays, n_samples]
        coeffs = alpha * self.cumprod_exclusive((1. - alpha)*phs_shift + 1e-10, dim = 1)
        amp_decay = 10**(-self.free_space_path_loss(z_vals, fc*1e9)/20)

        re_ch = torch.tanh(raw[..., 0])  # [N_dir, N_samples]
        im_ch = torch.tanh(raw[..., 1])     # [N_dir, N_samples]
        # Produce CFR
        sum_along_rays = torch.sum((re_ch + 1j*im_ch) * amp_decay * coeffs, dim=1)  # [n_rays]
        cfr = []
        last_cnt = 0
        for i in range(len(ray_batches)):
            cfr.append(torch.sum(sum_along_rays[last_cnt:last_cnt+ray_batches[i]], dim = 0))
            last_cnt += ray_batches[i]
        return torch.stack(cfr, dim = -1), depth, torch.abs(re_ch+1j*im_ch), weights

