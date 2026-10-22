import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_stages(stages):
    if stages is None:
        return set()
    if isinstance(stages, (list, tuple, set)):
        return {str(stage).strip().lower() for stage in stages}
    stages = str(stages).strip().lower()
    if stages in ("", "none", "add", "off", "false", "0"):
        return set()
    if stages == "all":
        return {"t1", "t2", "t3", "t4"}
    return {stage.strip() for stage in stages.split(",") if stage.strip()}


def normalize_experts(experts):
    if experts is None:
        experts = "add,dwt,gabor,dct"
    if isinstance(experts, str):
        experts = [item.strip().lower() for item in experts.split(",")]
    else:
        experts = [str(item).strip().lower() for item in experts]

    aliases = {
        "identity": "add",
        "spatial": "add",
        "none": "add",
        "lap": "laplacian",
    }
    allowed = {"add", "fft", "dct", "dwt", "laplacian", "gabor"}
    normalized = []
    for expert in experts:
        if not expert:
            continue
        expert = aliases.get(expert, expert)
        if expert not in allowed:
            raise ValueError("Unsupported frequency expert: %s" % expert)
        if expert not in normalized:
            normalized.append(expert)
    return normalized or ["add", "dwt", "gabor", "dct"]


def normalize_moe_prior(prior, experts):
    if prior is None:
        prior = "auto"
    if isinstance(prior, str):
        prior = prior.strip().lower()
        if prior in ("", "auto", "prior", "pg"):
            prior_map = {
                "add": 0.10,
                "dwt": 0.45,
                "gabor": 0.25,
                "dct": 0.20,
                "laplacian": 0.05,
                "fft": 0.05,
            }
        elif prior in ("uniform", "none", "false", "0"):
            prior_map = {expert: 1.0 for expert in experts}
        else:
            prior_map = {}
            for item in prior.split(","):
                if not item.strip():
                    continue
                name, value = item.split(":", 1)
                prior_map[name.strip().lower()] = float(value)
    elif isinstance(prior, dict):
        prior_map = {str(key).strip().lower(): float(value) for key, value in prior.items()}
    else:
        values = [float(value) for value in prior]
        if len(values) != len(experts):
            raise ValueError("Length of moe_prior must match frequency_experts.")
        prior_map = {expert: value for expert, value in zip(experts, values)}

    values = torch.tensor([max(prior_map.get(expert, 1e-6), 1e-6) for expert in experts], dtype=torch.float32)
    values = values / values.sum()
    return values


def ensure_even_hw(x):
    h, w = x.shape[-2:]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, h, w


def haar_dwt2(x):
    x, h, w = ensure_even_hw(x)
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (-a - b + c + d) * 0.5
    hl = (-a + b - c + d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, lh, hl, hh, (h, w)


def haar_idwt2(ll, lh, hl, hh, output_hw):
    a = (ll - lh - hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll + lh + hl + hh) * 0.5
    bsz, channels, h, w = ll.shape
    out = ll.new_zeros(bsz, channels, h * 2, w * 2)
    out[:, :, 0::2, 0::2] = a
    out[:, :, 0::2, 1::2] = b
    out[:, :, 1::2, 0::2] = c
    out[:, :, 1::2, 1::2] = d
    target_h, target_w = output_hw
    return out[:, :, :target_h, :target_w]


def dct_matrix(n, device, dtype):
    idx = torch.arange(n, device=device, dtype=dtype)
    k = idx[:, None]
    basis = torch.cos(math.pi / n * (idx[None, :] + 0.5) * k)
    basis[0, :] = basis[0, :] * math.sqrt(1.0 / n)
    if n > 1:
        basis[1:, :] = basis[1:, :] * math.sqrt(2.0 / n)
    return basis


def dct2(x):
    _, _, h, w = x.shape
    ch = dct_matrix(h, x.device, x.dtype)
    cw = dct_matrix(w, x.device, x.dtype)
    coeff = torch.einsum("kh,bchw->bckw", ch, x)
    coeff = torch.einsum("lw,bckw->bckl", cw, coeff)
    return coeff


def idct2(coeff):
    _, _, h, w = coeff.shape
    ch = dct_matrix(h, coeff.device, coeff.dtype)
    cw = dct_matrix(w, coeff.device, coeff.dtype)
    out = torch.einsum("kn,bckl->bcnl", ch, coeff)
    out = torch.einsum("lm,bcnl->bcnm", cw, out)
    return out


def frequency_mask(h, w, device, dtype, high_ratio=0.25, rfft=False):
    fy = torch.fft.fftfreq(h, device=device, dtype=dtype).abs()[:, None]
    if rfft:
        fx = torch.fft.rfftfreq(w, device=device, dtype=dtype).abs()[None, :]
    else:
        fx = torch.arange(w, device=device, dtype=dtype)[None, :] / max(w - 1, 1)
        fy = torch.arange(h, device=device, dtype=dtype)[:, None] / max(h - 1, 1)
    radius = torch.sqrt(fy * fy + fx * fx)
    cutoff = max(float(high_ratio), 1e-6)
    return (radius >= cutoff).to(dtype)


class FrequencyFusion(nn.Module):
    """
    Residual skip fusion with selectable frequency-domain branches.

    mode='add' exactly keeps the original spatial addition behavior.
    Other modes add a learnable residual initialized by gamma_init.
    """

    def __init__(
        self,
        channels,
        mode="add",
        use_gate=True,
        gamma_init=0.0,
        high_ratio=0.25,
        gabor_orientations=4,
        gabor_kernel_size=7,
        experts=None,
        moe_top_k=2,
        moe_temperature=1.0,
        moe_prior="auto",
        moe_balance_weight=0.0,
        moe_entropy_weight=0.0,
    ):
        super().__init__()
        self.channels = channels
        self.mode = str(mode).lower().replace("-", "_")
        if self.mode in ("moe", "pg_moe", "pg_afmoe"):
            self.mode = "afmoe"
        self.use_gate = use_gate
        self.high_ratio = high_ratio
        self.gabor_orientations = gabor_orientations
        self.gabor_kernel_size = gabor_kernel_size
        self.expert_names = normalize_experts(experts) if self.mode == "afmoe" else []
        self.num_experts = len(self.expert_names)
        self.moe_top_k = int(moe_top_k)
        self.moe_temperature = max(float(moe_temperature), 1e-6)
        self.moe_balance_weight = float(moe_balance_weight)
        self.moe_entropy_weight = float(moe_entropy_weight)
        self.last_moe_weights = None

        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        ) if use_gate and self.mode not in ("add", "afmoe") else None

        if self.mode == "afmoe":
            prior_values = normalize_moe_prior(moe_prior, self.expert_names)
            self.register_buffer("moe_prior", prior_values)
            hidden_channels = max(channels // 4, 8)
            self.router = nn.Sequential(
                nn.Conv2d(channels * 2, hidden_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, self.num_experts, kernel_size=1, bias=True),
            ) if use_gate else None
            if self.router is not None:
                nn.init.zeros_(self.router[-1].weight)
                with torch.no_grad():
                    self.router[-1].bias.copy_(torch.log(prior_values.clamp_min(1e-6)))
            self._init_moe_experts()
        elif self.mode == "dwt":
            self.low_fuse = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.high_fuse = nn.Sequential(
                nn.Conv2d(channels * 6, channels * 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels * 3),
                nn.ReLU(inplace=True),
            )
            self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        elif self.mode == "gabor":
            self.register_buffer("gabor_filters", self._build_gabor_filters())
            self.proj = nn.Sequential(
                nn.Conv2d(channels * gabor_orientations, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        elif self.mode in ("fft", "dct", "laplacian"):
            self.proj = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        elif self.mode == "add":
            self.proj = nn.Identity()
        else:
            raise ValueError("Unsupported fusion mode: %s" % mode)

    def _build_gabor_filters(self):
        kernel_size = int(self.gabor_kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        half = kernel_size // 2
        coords = torch.arange(-half, half + 1, dtype=torch.float32)
        try:
            yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        except TypeError:
            yy, xx = torch.meshgrid(coords, coords)
        filters = []
        sigma = max(kernel_size / 3.0, 1.0)
        wavelength = max(kernel_size / 2.0, 1.0)
        for i in range(int(self.gabor_orientations)):
            theta = math.pi * i / max(int(self.gabor_orientations), 1)
            x_theta = xx * math.cos(theta) + yy * math.sin(theta)
            y_theta = -xx * math.sin(theta) + yy * math.cos(theta)
            gaussian = torch.exp(-(x_theta.pow(2) + y_theta.pow(2)) / (2 * sigma * sigma))
            carrier = torch.cos(2 * math.pi * x_theta / wavelength)
            kernel = gaussian * carrier
            kernel = kernel - kernel.mean()
            kernel = kernel / (kernel.abs().sum() + 1e-6)
            filters.append(kernel)
        base = torch.stack(filters, dim=0).unsqueeze(1)
        return base.repeat(self.channels, 1, 1, 1)

    def _init_moe_experts(self):
        if "dwt" in self.expert_names:
            self.moe_low_fuse = nn.Sequential(
                nn.Conv2d(self.channels * 2, self.channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True),
            )
            self.moe_high_fuse = nn.Sequential(
                nn.Conv2d(self.channels * 6, self.channels * 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.channels * 3),
                nn.ReLU(inplace=True),
            )
            self.moe_dwt_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)

        if "gabor" in self.expert_names:
            self.register_buffer("moe_gabor_filters", self._build_gabor_filters())
            self.moe_gabor_proj = nn.Sequential(
                nn.Conv2d(self.channels * self.gabor_orientations, self.channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True),
            )

        if "dct" in self.expert_names:
            self.moe_dct_proj = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True),
            )

        if "fft" in self.expert_names:
            self.moe_fft_proj = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True),
            )

        if "laplacian" in self.expert_names:
            self.moe_laplacian_proj = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True),
            )

    def _apply_gate(self, freq_feat, x, skip):
        if self.gate is None:
            return freq_feat
        gate = self.gate(torch.cat([x, skip], dim=1))
        return freq_feat * gate

    def _fft_feature(self, x, skip):
        h, w = x.shape[-2:]
        fused_fft = torch.fft.rfft2(x + skip, norm="ortho")
        mask = frequency_mask(h, w, x.device, x.dtype, self.high_ratio, rfft=True)
        mask = mask[None, None, :, :]
        high_fft = fused_fft * mask
        return torch.fft.irfft2(high_fft, s=(h, w), norm="ortho")

    def _dct_feature(self, x, skip):
        coeff = dct2(x + skip)
        h, w = coeff.shape[-2:]
        mask = frequency_mask(h, w, coeff.device, coeff.dtype, self.high_ratio, rfft=False)
        return idct2(coeff * mask[None, None, :, :])

    def _dwt_feature(self, x, skip):
        x_ll, x_lh, x_hl, x_hh, output_hw = haar_dwt2(x)
        s_ll, s_lh, s_hl, s_hh, _ = haar_dwt2(skip)
        ll = self.low_fuse(torch.cat([x_ll, s_ll], dim=1))
        high = self.high_fuse(torch.cat([x_lh, s_lh, x_hl, s_hl, x_hh, s_hh], dim=1))
        lh, hl, hh = torch.chunk(high, 3, dim=1)
        return self.proj(haar_idwt2(ll, lh, hl, hh, output_hw))

    def _laplacian_feature(self, x, skip):
        fused = x + skip
        low = F.avg_pool2d(fused, kernel_size=3, stride=1, padding=1)
        low = F.avg_pool2d(low, kernel_size=3, stride=1, padding=1)
        return fused - low

    def _gabor_feature(self, x, skip):
        fused = x + skip
        filters = self.gabor_filters.to(device=fused.device, dtype=fused.dtype)
        feat = F.conv2d(fused, filters, padding=filters.shape[-1] // 2, groups=self.channels)
        return self.proj(feat)

    def _moe_dwt_feature(self, x, skip):
        x_ll, x_lh, x_hl, x_hh, output_hw = haar_dwt2(x)
        s_ll, s_lh, s_hl, s_hh, _ = haar_dwt2(skip)
        ll = self.moe_low_fuse(torch.cat([x_ll, s_ll], dim=1))
        high = self.moe_high_fuse(torch.cat([x_lh, s_lh, x_hl, s_hl, x_hh, s_hh], dim=1))
        lh, hl, hh = torch.chunk(high, 3, dim=1)
        return self.moe_dwt_proj(haar_idwt2(ll, lh, hl, hh, output_hw))

    def _moe_gabor_feature(self, x, skip):
        fused = x + skip
        filters = self.moe_gabor_filters.to(device=fused.device, dtype=fused.dtype)
        feat = F.conv2d(fused, filters, padding=filters.shape[-1] // 2, groups=self.channels)
        return self.moe_gabor_proj(feat)

    def _moe_expert_feature(self, expert, x, skip):
        if expert == "add":
            return torch.zeros_like(x)
        if expert == "dwt":
            return self._moe_dwt_feature(x, skip)
        if expert == "gabor":
            return self._moe_gabor_feature(x, skip)
        if expert == "dct":
            return self.moe_dct_proj(self._dct_feature(x, skip))
        if expert == "fft":
            return self.moe_fft_proj(self._fft_feature(x, skip))
        if expert == "laplacian":
            return self.moe_laplacian_proj(self._laplacian_feature(x, skip))
        raise ValueError("Unsupported frequency expert: %s" % expert)

    def _topk_softmax(self, logits):
        scaled = logits / self.moe_temperature
        if self.moe_top_k <= 0 or self.moe_top_k >= self.num_experts:
            return torch.softmax(scaled, dim=1)
        values, indices = torch.topk(scaled, k=self.moe_top_k, dim=1)
        sparse_logits = scaled.new_full(scaled.shape, -1e4)
        sparse_logits.scatter_(1, indices, values)
        return torch.softmax(sparse_logits, dim=1)

    def _afmoe_feature(self, x, skip):
        if self.router is None:
            weights = self.moe_prior.to(device=x.device, dtype=x.dtype)[None, :, None, None]
            weights = weights.expand(x.shape[0], -1, x.shape[-2], x.shape[-1])
        else:
            logits = self.router(torch.cat([x, skip], dim=1))
            weights = self._topk_softmax(logits)
        self.last_moe_weights = weights

        expert_outputs = [
            self._moe_expert_feature(expert, x, skip)
            for expert in self.expert_names
        ]
        stacked = torch.stack(expert_outputs, dim=1)
        return (weights.unsqueeze(2) * stacked).sum(dim=1)

    def get_moe_loss(self):
        if self.mode != "afmoe" or self.last_moe_weights is None:
            return None
        if self.moe_balance_weight <= 0 and self.moe_entropy_weight <= 0:
            return None

        weights = self.last_moe_weights
        loss = weights.new_tensor(0.0)
        if self.moe_balance_weight > 0:
            target = self.moe_prior.to(device=weights.device, dtype=weights.dtype)
            mean_load = weights.mean(dim=(0, 2, 3))
            loss = loss + self.moe_balance_weight * F.mse_loss(mean_load, target)
        if self.moe_entropy_weight > 0:
            safe_weights = weights.clamp_min(1e-8)
            entropy = -(safe_weights * safe_weights.log()).sum(dim=1).mean()
            loss = loss + self.moe_entropy_weight * entropy
        return loss

    def forward(self, x, skip):
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)

        spatial = x + skip
        if self.mode == "add":
            return spatial
        if self.mode == "afmoe":
            freq_feat = self._afmoe_feature(x, skip)
            return spatial + self.gamma * freq_feat

        if self.mode == "fft":
            freq_feat = self.proj(self._fft_feature(x, skip))
        elif self.mode == "dct":
            freq_feat = self.proj(self._dct_feature(x, skip))
        elif self.mode == "dwt":
            freq_feat = self._dwt_feature(x, skip)
        elif self.mode == "laplacian":
            freq_feat = self.proj(self._laplacian_feature(x, skip))
        elif self.mode == "gabor":
            freq_feat = self._gabor_feature(x, skip)
        else:
            raise ValueError("Unsupported fusion mode: %s" % self.mode)

        freq_feat = self._apply_gate(freq_feat, x, skip)
        return spatial + self.gamma * freq_feat
