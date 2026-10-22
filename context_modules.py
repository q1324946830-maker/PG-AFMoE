import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_rates(rates):
    if rates is None:
        return (1, 2, 3)
    if isinstance(rates, str):
        rates = [item.strip() for item in rates.split(",") if item.strip()]
    return tuple(max(int(rate), 1) for rate in rates)


class IdentityContext(nn.Module):
    def forward(self, x):
        return x


class LightASPP(nn.Module):
    """
    Compact DeepLabV3-style ASPP block for the bottleneck feature map.
    """

    def __init__(self, channels, rates=(1, 2, 3), dropout=0.0):
        super().__init__()
        rates = parse_rates(rates)
        branch_channels = max(channels // 4, 16)

        branches = [
            nn.Sequential(
                nn.Conv2d(channels, branch_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in rates:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        branch_channels,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

        fuse_layers = [
            nn.Conv2d(branch_channels * (len(branches) + 1), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        if dropout and dropout > 0:
            fuse_layers.append(nn.Dropout2d(float(dropout)))
        self.fuse = nn.Sequential(*fuse_layers)

    def forward(self, x):
        size = x.shape[-2:]
        features = [branch(x) for branch in self.branches]
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
        features.append(pooled)
        return self.fuse(torch.cat(features, dim=1))


def build_context_module(name, channels, rates="1,2,3", dropout=0.0):
    name = "none" if name is None else str(name).strip().lower()
    if name in ("", "none", "identity", "off", "false", "0"):
        return IdentityContext()
    if name in ("aspp", "light_aspp", "deeplabv3"):
        return LightASPP(channels, rates=parse_rates(rates), dropout=dropout)
    raise ValueError("Unsupported context_module: %s" % name)
