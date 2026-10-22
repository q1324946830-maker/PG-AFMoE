import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    lovasz_hinge = None

__all__ = [
    "BCEDiceLoss",
    "FocalBCEDiceLoss",
    "LovaszHingeLoss",
    "BCEDiceWithGeometryLoss",
    "FocalBCEDiceWithGeometryLoss",
]


def unpack_seg_output(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = unpack_seg_output(input)
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        prob = torch.sigmoid(input)
        num = target.size(0)
        prob = prob.view(num, -1)
        target = target.view(num, -1)
        intersection = prob * target
        dice = (2.0 * intersection.sum(1) + smooth) / (prob.sum(1) + target.sum(1) + smooth)
        dice = 1.0 - dice.sum() / num
        return 0.5 * bce + dice


class FocalBCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, hard_pixel_ratio=0.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.hard_pixel_ratio = float(hard_pixel_ratio)

    def _reduce_focal(self, focal):
        if self.hard_pixel_ratio <= 0:
            return focal.mean()
        flat = focal.reshape(-1)
        k = max(int(flat.numel() * self.hard_pixel_ratio), 1)
        k = min(k, flat.numel())
        return torch.topk(flat, k=k, largest=True).values.mean()

    def forward(self, input, target):
        input = unpack_seg_output(input)
        bce = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        prob = torch.sigmoid(input)
        pt = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal = alpha_t * torch.pow(1.0 - pt, self.gamma) * bce
        focal = self._reduce_focal(focal)

        smooth = 1e-5
        num = target.size(0)
        prob = prob.view(num, -1)
        target = target.view(num, -1)
        intersection = prob * target
        dice = (2.0 * intersection.sum(1) + smooth) / (prob.sum(1) + target.sum(1) + smooth)
        dice = 1.0 - dice.sum() / num
        return 0.5 * focal + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        if lovasz_hinge is None:
            raise ImportError("LovaszSoftmax is required for LovaszHingeLoss.")
        input = unpack_seg_output(input).squeeze(1)
        target = target.squeeze(1)
        return lovasz_hinge(input, target, per_image=True)


def wasserstein_diversity_loss(centers):
    k, _ = centers.shape
    if k <= 1:
        return torch.tensor(0.0, device=centers.device)

    mu = torch.mean(centers, dim=0)
    centered = centers - mu
    covariance = (centered.t() @ centered) / k
    term_mean = torch.sum(mu ** 2)
    term_trace = torch.trace(covariance)

    try:
        eigenvalues = torch.linalg.eigvalsh(covariance)
        term_trace_sqrt = torch.sum(torch.sqrt(torch.clamp(eigenvalues, min=0)))
    except torch.linalg.LinAlgError:
        return torch.tensor(0.0, device=centers.device)

    w2_squared = term_mean + term_trace - 2.0 * term_trace_sqrt
    return torch.clamp(w2_squared, min=0)


class BCEDiceWithGeometryLoss(nn.Module):
    requires_model = True

    def __init__(self, div_weight=0.1, scale_weight=0.1):
        super().__init__()
        self.bce_dice = BCEDiceLoss()
        self.div_weight = div_weight
        self.scale_weight = scale_weight

    def calculate_scale_loss(self, att, dif, log_sigma):
        if not log_sigma.requires_grad:
            return torch.tensor(0.0, device=att.device)

        with torch.no_grad():
            weight = att.detach().unsqueeze(-1)
            samples_per_ball = weight.sum(dim=1, keepdim=True) + 1e-6

        diff_elem_sq = dif.pow(2)
        observed_var_batched = (weight * diff_elem_sq).sum(dim=1, keepdim=True) / samples_per_ball
        observed_var = observed_var_batched.mean(dim=0).squeeze(0)
        learned_var = F.softplus(log_sigma).pow(2)
        return F.mse_loss(observed_var.detach(), learned_var)

    def forward(self, model_output, target, model):
        if isinstance(model_output, tuple):
            seg_output, loss_intermediates = model_output
        else:
            seg_output = model_output
            loss_intermediates = None

        main_loss = self.bce_dice(seg_output, target)

        if isinstance(model, torch.nn.DataParallel):
            base_model = model.module
        else:
            base_model = model

        gbc_module = getattr(base_model, "gbc", None)
        div_loss = main_loss.new_tensor(0.0)
        if gbc_module is not None:
            div_loss = wasserstein_diversity_loss(gbc_module.centers)

        scale_loss = main_loss.new_tensor(0.0)
        if (gbc_module is not None and loss_intermediates is not None
                and self.scale_weight > 0 and gbc_module.use_diag_cov):
            log_sigma = gbc_module.log_sigma
            scale_loss_1 = self.calculate_scale_loss(
                loss_intermediates["att_1"],
                loss_intermediates["dif_1"],
                log_sigma,
            )
            scale_loss_2 = self.calculate_scale_loss(
                loss_intermediates["att_2"],
                loss_intermediates["dif_2"],
                log_sigma,
            )
            scale_loss = (scale_loss_1 + scale_loss_2) / 2.0

        total_loss = main_loss + self.div_weight * div_loss + self.scale_weight * scale_loss
        if hasattr(base_model, "get_moe_aux_loss"):
            moe_loss = base_model.get_moe_aux_loss()
            if moe_loss is not None:
                total_loss = total_loss + moe_loss

        return total_loss


class FocalBCEDiceWithGeometryLoss(BCEDiceWithGeometryLoss):
    requires_model = True

    def __init__(
        self,
        div_weight=0.1,
        scale_weight=0.1,
        focal_alpha=0.25,
        focal_gamma=2.0,
        hard_pixel_ratio=0.0,
    ):
        super().__init__(div_weight=div_weight, scale_weight=scale_weight)
        self.bce_dice = FocalBCEDiceLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            hard_pixel_ratio=hard_pixel_ratio,
        )
