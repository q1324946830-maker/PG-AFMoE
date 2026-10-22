import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAlignModule(nn.Module):
    """
    Lightweight SFNet-style spatial alignment for encoder-decoder skip fusion.

    The last offset layer is zero-initialized, so enabling the module starts from
    an identity warp and learns small residual displacements during training.
    """

    def __init__(self, channels, hidden_channels=None, max_flow=4.0):
        super().__init__()
        hidden_channels = hidden_channels or max(channels // 4, 8)
        self.max_flow = float(max_flow)
        self.offset = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1, bias=True),
        )
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)

    def forward(self, x, skip):
        if skip.shape[-2:] != x.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)

        flow = torch.tanh(self.offset(torch.cat([x, skip], dim=1))) * self.max_flow
        bsz, _, height, width = x.shape

        y = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
        x_coord = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype)
        try:
            yy, xx = torch.meshgrid(y, x_coord, indexing="ij")
        except TypeError:
            yy, xx = torch.meshgrid(y, x_coord)
        base_grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(bsz, height, width, 2)

        norm_x = flow[:, 0] * (2.0 / max(width - 1, 1))
        norm_y = flow[:, 1] * (2.0 / max(height - 1, 1))
        grid = base_grid + torch.stack((norm_x, norm_y), dim=-1)

        return F.grid_sample(skip, grid, mode="bilinear", padding_mode="border", align_corners=True)


SFGA = SpatialAlignModule
