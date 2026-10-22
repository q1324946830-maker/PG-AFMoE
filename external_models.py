import importlib.util
import hashlib
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from alignment_fusion import SFGA
from context_modules import build_context_module
from frequency_fusion import FrequencyFusion, normalize_stages


EXTERNAL_ARCH_NAMES = [
    "UNet",
    "UNetPlusPlus",
    "UNet3Plus",
    "MALUNet",
    "EGEUNet",
]

__all__ = [
    *EXTERNAL_ARCH_NAMES,
    "build_external_model",
]


def _load_module(module_name, file_path, package_paths=None):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(file_path),
        submodule_search_locations=package_paths,
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load external model module: {}".format(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _related_root(root=None):
    return Path(root or os.environ.get("RELATED_MODELS_ROOT", "../Related-Models")).expanduser()


def _module_token(root):
    value = str(_related_root(root).resolve()).encode("utf-8")
    return hashlib.sha1(value).hexdigest()[:10]


def _load_unet_class(root):
    root = _related_root(root)
    package_dir = root / "Pytorch-UNet-master" / "unet"
    package_name = "related_pytorch_unet_{}".format(_module_token(root))
    package = _load_module(
        package_name,
        package_dir / "__init__.py",
        [str(package_dir)],
    )
    return package.UNet


def _load_unet3_class(root):
    root = _related_root(root)
    source_dir = root / "UNetPlusPlusPlus-master" / "models"
    module_name = "related_unet3plus_{}".format(_module_token(root))
    if module_name in sys.modules:
        return sys.modules[module_name].UNet_3Plus
    sys.path.insert(0, str(source_dir))
    try:
        module = _load_module(module_name, source_dir / "UNet_3Plus.py")
    finally:
        if sys.path and sys.path[0] == str(source_dir):
            sys.path.pop(0)
    return module.UNet_3Plus


def _load_mal_class(root):
    root = _related_root(root)
    module = _load_module(
        "related_malunet_{}".format(_module_token(root)),
        root / "MALUNet-main" / "models" / "malunet.py",
    )
    return module.MALUNet


def _load_ege_class(root):
    root = _related_root(root)
    module = _load_module(
        "related_egeunet_{}".format(_module_token(root)),
        root / "EGE-UNet-main" / "models" / "egeunet.py",
    )
    return module.EGEUNet


class GraftMixin:
    @staticmethod
    def _configured_stages(stages, available_stages):
        normalized = normalize_stages(stages)
        if "all" in normalized or str(stages).strip().lower() == "all":
            return set(available_stages)
        return normalized

    def _init_graft(
        self,
        stage_channels,
        bottleneck_channels,
        frequency_mode="add",
        frequency_stages="none",
        frequency_gate=True,
        frequency_gamma=0.0,
        frequency_high_ratio=0.25,
        frequency_experts="add,dwt,gabor,dct",
        moe_top_k=2,
        moe_temperature=1.0,
        moe_prior="auto",
        moe_balance_weight=0.0,
        moe_entropy_weight=0.0,
        use_align=False,
        align_stages="none",
        align_max_flow=4.0,
        context_module="none",
        context_rates="1,2,3",
        context_dropout=0.0,
    ):
        self.graft_frequency_mode = str(frequency_mode).lower()
        available_stages = set(stage_channels)
        self.graft_frequency_stages = self._configured_stages(
            frequency_stages,
            available_stages,
        )
        self.graft_align_stages = (
            self._configured_stages(align_stages, available_stages)
            if use_align else set()
        )
        self.graft_fusions = nn.ModuleDict()
        self.graft_aligners = nn.ModuleDict()

        for stage, channels in stage_channels.items():
            if stage in self.graft_frequency_stages:
                self.graft_fusions[stage] = FrequencyFusion(
                    channels=channels,
                    mode=frequency_mode,
                    use_gate=frequency_gate,
                    gamma_init=frequency_gamma,
                    high_ratio=frequency_high_ratio,
                    experts=frequency_experts,
                    moe_top_k=moe_top_k,
                    moe_temperature=moe_temperature,
                    moe_prior=moe_prior,
                    moe_balance_weight=moe_balance_weight,
                    moe_entropy_weight=moe_entropy_weight,
                )
            if stage in self.graft_align_stages:
                self.graft_aligners[stage] = SFGA(
                    channels,
                    max_flow=align_max_flow,
                )

        self.graft_context = build_context_module(
            context_module,
            bottleneck_channels,
            rates=context_rates,
            dropout=context_dropout,
        )

    def _graft_prepare_skip(self, stage, x, skip):
        if skip.shape[-2:] != x.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if stage in self.graft_aligners:
            skip = self.graft_aligners[stage](x, skip)
        return skip

    def _graft_skip(self, stage, x, skip):
        skip = self._graft_prepare_skip(stage, x, skip)
        if stage not in self.graft_fusions:
            return skip
        return self.graft_fusions[stage](x, skip) - x

    def _graft_add(self, stage, x, skip):
        skip = self._graft_prepare_skip(stage, x, skip)
        if stage not in self.graft_fusions:
            return x + skip
        return self.graft_fusions[stage](x, skip)

    def _graft_context(self, x):
        return self.graft_context(x)

    def get_moe_aux_loss(self):
        losses = []
        for fusion in self.graft_fusions.values():
            loss = fusion.get_moe_loss()
            if loss is not None:
                losses.append(loss)
        return sum(losses) / len(losses) if losses else None


class UNet(GraftMixin, nn.Module):
    def __init__(
        self,
        num_classes=1,
        input_channels=3,
        deep_supervision=False,
        related_models_root=None,
        frequency_mode="add",
        frequency_stages="none",
        frequency_gate=True,
        frequency_gamma=0.0,
        frequency_high_ratio=0.25,
        frequency_experts="add,dwt,gabor,dct",
        moe_top_k=2,
        moe_temperature=1.0,
        moe_prior="auto",
        moe_balance_weight=0.0,
        moe_entropy_weight=0.0,
        use_align=False,
        align_stages="none",
        align_max_flow=4.0,
        context_module="none",
        context_rates="1,2,3",
        context_dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.related_models_root = str(_related_root(related_models_root))
        self.backbone = _load_unet_class(self.related_models_root)(
            n_channels=input_channels,
            n_classes=num_classes,
            bilinear=False,
        )
        self.n_channels = input_channels
        self.n_classes = num_classes
        self._init_graft(
            {"t1": 64, "t2": 128, "t3": 256, "t4": 512},
            1024,
            frequency_mode,
            frequency_stages,
            frequency_gate,
            frequency_gamma,
            frequency_high_ratio,
            frequency_experts,
            moe_top_k,
            moe_temperature,
            moe_prior,
            moe_balance_weight,
            moe_entropy_weight,
            use_align,
            align_stages,
            align_max_flow,
            context_module,
            context_rates,
            context_dropout,
        )

    def _up(self, block, x, skip, stage):
        x = block.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        skip = self._graft_skip(stage, x, skip)
        return block.conv(torch.cat([skip, x], dim=1))

    def forward(self, x):
        backbone = self.backbone
        x1 = backbone.inc(x)
        x2 = backbone.down1(x1)
        x3 = backbone.down2(x2)
        x4 = backbone.down3(x3)
        x5 = self._graft_context(backbone.down4(x4))
        x = self._up(backbone.up1, x5, x4, "t4")
        x = self._up(backbone.up2, x, x3, "t3")
        x = self._up(backbone.up3, x, x2, "t2")
        x = self._up(backbone.up4, x, x1, "t1")
        return backbone.outc(x)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _UpProject(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x, target):
        x = self.project(x)
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)


class UNetPlusPlus(GraftMixin, nn.Module):
    def __init__(
        self,
        num_classes=1,
        input_channels=3,
        deep_supervision=False,
        base_channels=32,
        frequency_mode="add",
        frequency_stages="none",
        frequency_gate=True,
        frequency_gamma=0.0,
        frequency_high_ratio=0.25,
        frequency_experts="add,dwt,gabor,dct",
        moe_top_k=2,
        moe_temperature=1.0,
        moe_prior="auto",
        moe_balance_weight=0.0,
        moe_entropy_weight=0.0,
        use_align=False,
        align_stages="none",
        align_max_flow=4.0,
        context_module="none",
        context_rates="1,2,3",
        context_dropout=0.0,
        **kwargs
    ):
        super().__init__()
        channels = [base_channels * (2 ** i) for i in range(5)]
        self.pool = nn.MaxPool2d(2)
        self.enc0 = _ConvBlock(input_channels, channels[0])
        self.enc1 = _ConvBlock(channels[0], channels[1])
        self.enc2 = _ConvBlock(channels[1], channels[2])
        self.enc3 = _ConvBlock(channels[2], channels[3])
        self.enc4 = _ConvBlock(channels[3], channels[4])

        self.ups = nn.ModuleDict({
            "x01": _UpProject(channels[1], channels[0]),
            "x11": _UpProject(channels[2], channels[1]),
            "x21": _UpProject(channels[3], channels[2]),
            "x31": _UpProject(channels[4], channels[3]),
            "x02": _UpProject(channels[1], channels[0]),
            "x12": _UpProject(channels[2], channels[1]),
            "x22": _UpProject(channels[3], channels[2]),
            "x03": _UpProject(channels[1], channels[0]),
            "x13": _UpProject(channels[2], channels[1]),
            "x04": _UpProject(channels[1], channels[0]),
        })
        self.x01 = _ConvBlock(channels[0] * 2, channels[0])
        self.x11 = _ConvBlock(channels[1] * 2, channels[1])
        self.x21 = _ConvBlock(channels[2] * 2, channels[2])
        self.x31 = _ConvBlock(channels[3] * 2, channels[3])
        self.x02 = _ConvBlock(channels[0] * 3, channels[0])
        self.x12 = _ConvBlock(channels[1] * 3, channels[1])
        self.x22 = _ConvBlock(channels[2] * 3, channels[2])
        self.x03 = _ConvBlock(channels[0] * 4, channels[0])
        self.x13 = _ConvBlock(channels[1] * 4, channels[1])
        self.x04 = _ConvBlock(channels[0] * 5, channels[0])
        self.final = nn.Conv2d(channels[0], num_classes, 1)
        self._init_graft(
            {"t1": channels[0], "t2": channels[1], "t3": channels[2], "t4": channels[3]},
            channels[4],
            frequency_mode,
            frequency_stages,
            frequency_gate,
            frequency_gamma,
            frequency_high_ratio,
            frequency_experts,
            moe_top_k,
            moe_temperature,
            moe_prior,
            moe_balance_weight,
            moe_entropy_weight,
            use_align,
            align_stages,
            align_max_flow,
            context_module,
            context_rates,
            context_dropout,
        )

    def _node(self, stage, anchor, x, previous, block, up):
        x = up(x, anchor)
        anchor = self._graft_skip(stage, x, anchor)
        return block(torch.cat([anchor, *previous, x], dim=1))

    def forward(self, x):
        x00 = self.enc0(x)
        x10 = self.enc1(self.pool(x00))
        x20 = self.enc2(self.pool(x10))
        x30 = self.enc3(self.pool(x20))
        x40 = self._graft_context(self.enc4(self.pool(x30)))

        x01 = self._node("t1", x00, x10, [], self.x01, self.ups["x01"])
        x11 = self._node("t2", x10, x20, [], self.x11, self.ups["x11"])
        x21 = self._node("t3", x20, x30, [], self.x21, self.ups["x21"])
        x31 = self._node("t4", x30, x40, [], self.x31, self.ups["x31"])
        x02 = self._node("t1", x00, x11, [x01], self.x02, self.ups["x02"])
        x12 = self._node("t2", x10, x21, [x11], self.x12, self.ups["x12"])
        x22 = self._node("t3", x20, x31, [x21], self.x22, self.ups["x22"])
        x03 = self._node("t1", x00, x12, [x01, x02], self.x03, self.ups["x03"])
        x13 = self._node("t2", x10, x22, [x11, x12], self.x13, self.ups["x13"])
        x04 = self._node("t1", x00, x13, [x01, x02, x03], self.x04, self.ups["x04"])
        return self.final(x04)


class UNet3Plus(GraftMixin, nn.Module):
    def __init__(
        self,
        num_classes=1,
        input_channels=3,
        deep_supervision=False,
        related_models_root=None,
        frequency_mode="add",
        frequency_stages="none",
        frequency_gate=True,
        frequency_gamma=0.0,
        frequency_high_ratio=0.25,
        frequency_experts="add,dwt,gabor,dct",
        moe_top_k=2,
        moe_temperature=1.0,
        moe_prior="auto",
        moe_balance_weight=0.0,
        moe_entropy_weight=0.0,
        use_align=False,
        align_stages="none",
        align_max_flow=4.0,
        context_module="none",
        context_rates="1,2,3",
        context_dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.related_models_root = str(_related_root(related_models_root))
        self.backbone = _load_unet3_class(self.related_models_root)(
            in_channels=input_channels,
            n_classes=num_classes,
        )
        self._init_graft(
            {"t1": 64, "t2": 64, "t3": 64, "t4": 64},
            1024,
            frequency_mode,
            frequency_stages,
            frequency_gate,
            frequency_gamma,
            frequency_high_ratio,
            frequency_experts,
            moe_top_k,
            moe_temperature,
            moe_prior,
            moe_balance_weight,
            moe_entropy_weight,
            use_align,
            align_stages,
            align_max_flow,
            context_module,
            context_rates,
            context_dropout,
        )

    def _full_scale(self, stage, features):
        if len(features) > 1:
            anchor = features[0]
            features = [anchor] + [self._graft_skip(stage, anchor, feature) for feature in features[1:]]
        return torch.cat(features, dim=1)

    def forward(self, inputs):
        backbone = self.backbone
        h1 = backbone.conv1(inputs)
        h2 = backbone.conv2(backbone.maxpool1(h1))
        h3 = backbone.conv3(backbone.maxpool2(h2))
        h4 = backbone.conv4(backbone.maxpool3(h3))
        hd5 = self._graft_context(backbone.conv5(backbone.maxpool4(h4)))

        h1_pt_hd4 = backbone.h1_PT_hd4_relu(backbone.h1_PT_hd4_bn(backbone.h1_PT_hd4_conv(backbone.h1_PT_hd4(h1))))
        h2_pt_hd4 = backbone.h2_PT_hd4_relu(backbone.h2_PT_hd4_bn(backbone.h2_PT_hd4_conv(backbone.h2_PT_hd4(h2))))
        h3_pt_hd4 = backbone.h3_PT_hd4_relu(backbone.h3_PT_hd4_bn(backbone.h3_PT_hd4_conv(backbone.h3_PT_hd4(h3))))
        h4_cat_hd4 = backbone.h4_Cat_hd4_relu(backbone.h4_Cat_hd4_bn(backbone.h4_Cat_hd4_conv(h4)))
        hd5_ut_hd4 = backbone.hd5_UT_hd4_relu(backbone.hd5_UT_hd4_bn(backbone.hd5_UT_hd4_conv(backbone.hd5_UT_hd4(hd5))))
        hd4 = backbone.relu4d_1(backbone.bn4d_1(backbone.conv4d_1(self._full_scale("t4", [
            h1_pt_hd4, h2_pt_hd4, h3_pt_hd4, h4_cat_hd4, hd5_ut_hd4]))))

        h1_pt_hd3 = backbone.h1_PT_hd3_relu(backbone.h1_PT_hd3_bn(backbone.h1_PT_hd3_conv(backbone.h1_PT_hd3(h1))))
        h2_pt_hd3 = backbone.h2_PT_hd3_relu(backbone.h2_PT_hd3_bn(backbone.h2_PT_hd3_conv(backbone.h2_PT_hd3(h2))))
        h3_cat_hd3 = backbone.h3_Cat_hd3_relu(backbone.h3_Cat_hd3_bn(backbone.h3_Cat_hd3_conv(h3)))
        hd4_ut_hd3 = backbone.hd4_UT_hd3_relu(backbone.hd4_UT_hd3_bn(backbone.hd4_UT_hd3_conv(backbone.hd4_UT_hd3(hd4))))
        hd5_ut_hd3 = backbone.hd5_UT_hd3_relu(backbone.hd5_UT_hd3_bn(backbone.hd5_UT_hd3_conv(backbone.hd5_UT_hd3(hd5))))
        hd3 = backbone.relu3d_1(backbone.bn3d_1(backbone.conv3d_1(self._full_scale("t3", [
            h1_pt_hd3, h2_pt_hd3, h3_cat_hd3, hd4_ut_hd3, hd5_ut_hd3]))))

        h1_pt_hd2 = backbone.h1_PT_hd2_relu(backbone.h1_PT_hd2_bn(backbone.h1_PT_hd2_conv(backbone.h1_PT_hd2(h1))))
        h2_cat_hd2 = backbone.h2_Cat_hd2_relu(backbone.h2_Cat_hd2_bn(backbone.h2_Cat_hd2_conv(h2)))
        hd3_ut_hd2 = backbone.hd3_UT_hd2_relu(backbone.hd3_UT_hd2_bn(backbone.hd3_UT_hd2_conv(backbone.hd3_UT_hd2(hd3))))
        hd4_ut_hd2 = backbone.hd4_UT_hd2_relu(backbone.hd4_UT_hd2_bn(backbone.hd4_UT_hd2_conv(backbone.hd4_UT_hd2(hd4))))
        hd5_ut_hd2 = backbone.hd5_UT_hd2_relu(backbone.hd5_UT_hd2_bn(backbone.hd5_UT_hd2_conv(backbone.hd5_UT_hd2(hd5))))
        hd2 = backbone.relu2d_1(backbone.bn2d_1(backbone.conv2d_1(self._full_scale("t2", [
            h1_pt_hd2, h2_cat_hd2, hd3_ut_hd2, hd4_ut_hd2, hd5_ut_hd2]))))

        h1_cat_hd1 = backbone.h1_Cat_hd1_relu(backbone.h1_Cat_hd1_bn(backbone.h1_Cat_hd1_conv(h1)))
        hd2_ut_hd1 = backbone.hd2_UT_hd1_relu(backbone.hd2_UT_hd1_bn(backbone.hd2_UT_hd1_conv(backbone.hd2_UT_hd1(hd2))))
        hd3_ut_hd1 = backbone.hd3_UT_hd1_relu(backbone.hd3_UT_hd1_bn(backbone.hd3_UT_hd1_conv(backbone.hd3_UT_hd1(hd3))))
        hd4_ut_hd1 = backbone.hd4_UT_hd1_relu(backbone.hd4_UT_hd1_bn(backbone.hd4_UT_hd1_conv(backbone.hd4_UT_hd1(hd4))))
        hd5_ut_hd1 = backbone.hd5_UT_hd1_relu(backbone.hd5_UT_hd1_bn(backbone.hd5_UT_hd1_conv(backbone.hd5_UT_hd1(hd5))))
        hd1 = backbone.relu1d_1(backbone.bn1d_1(backbone.conv1d_1(self._full_scale("t1", [
            h1_cat_hd1, hd2_ut_hd1, hd3_ut_hd1, hd4_ut_hd1, hd5_ut_hd1]))))
        return backbone.outconv1(hd1)


class MALUNet(GraftMixin, nn.Module):
    def __init__(
        self,
        num_classes=1,
        input_channels=3,
        deep_supervision=False,
        c_list=None,
        split_att="fc",
        bridge=True,
        frequency_mode="add",
        frequency_stages="none",
        frequency_gate=True,
        frequency_gamma=0.0,
        frequency_high_ratio=0.25,
        frequency_experts="add,dwt,gabor,dct",
        moe_top_k=2,
        moe_temperature=1.0,
        moe_prior="auto",
        moe_balance_weight=0.0,
        moe_entropy_weight=0.0,
        use_align=False,
        align_stages="none",
        align_max_flow=4.0,
        context_module="none",
        context_rates="1,2,3",
        context_dropout=0.0,
        related_models_root=None,
        **kwargs
    ):
        c_list = c_list or [8, 16, 24, 32, 48, 64]
        super().__init__()
        self.related_models_root = str(_related_root(related_models_root))
        self.backbone = _load_mal_class(self.related_models_root)(
            num_classes=num_classes,
            input_channels=input_channels,
            c_list=c_list,
            split_att=split_att,
            bridge=bridge,
        )
        self._init_graft(
            {"t1": c_list[0], "t2": c_list[1], "t3": c_list[2], "t4": c_list[3], "t5": c_list[4]},
            c_list[5],
            frequency_mode,
            frequency_stages,
            frequency_gate,
            frequency_gamma,
            frequency_high_ratio,
            frequency_experts,
            moe_top_k,
            moe_temperature,
            moe_prior,
            moe_balance_weight,
            moe_entropy_weight,
            use_align,
            align_stages,
            align_max_flow,
            context_module,
            context_rates,
            context_dropout,
        )

    def forward(self, x):
        backbone = self.backbone
        out = F.gelu(F.max_pool2d(backbone.ebn1(backbone.encoder1(x)), 2, 2))
        t1 = out
        out = F.gelu(F.max_pool2d(backbone.ebn2(backbone.encoder2(out)), 2, 2))
        t2 = out
        out = F.gelu(F.max_pool2d(backbone.ebn3(backbone.encoder3(out)), 2, 2))
        t3 = out
        out = F.gelu(F.max_pool2d(backbone.ebn4(backbone.encoder4(out)), 2, 2))
        t4 = out
        out = F.gelu(F.max_pool2d(backbone.ebn5(backbone.encoder5(out)), 2, 2))
        t5 = out
        if backbone.bridge:
            t1, t2, t3, t4, t5 = backbone.scab(t1, t2, t3, t4, t5)
        out = self._graft_context(F.gelu(backbone.encoder6(out)))

        out5 = F.gelu(backbone.dbn1(backbone.decoder1(out)))
        out5 = self._graft_add("t5", out5, t5)
        out4 = F.gelu(F.interpolate(backbone.dbn2(backbone.decoder2(out5)), scale_factor=2, mode="bilinear", align_corners=False))
        out4 = self._graft_add("t4", out4, t4)
        out3 = F.gelu(F.interpolate(backbone.dbn3(backbone.decoder3(out4)), scale_factor=2, mode="bilinear", align_corners=False))
        out3 = self._graft_add("t3", out3, t3)
        out2 = F.gelu(F.interpolate(backbone.dbn4(backbone.decoder4(out3)), scale_factor=2, mode="bilinear", align_corners=False))
        out2 = self._graft_add("t2", out2, t2)
        out1 = F.gelu(F.interpolate(backbone.dbn5(backbone.decoder5(out2)), scale_factor=2, mode="bilinear", align_corners=False))
        out1 = self._graft_add("t1", out1, t1)
        return F.interpolate(backbone.final(out1), scale_factor=2, mode="bilinear", align_corners=False)


class EGEUNet(GraftMixin, nn.Module):
    def __init__(
        self,
        num_classes=1,
        input_channels=3,
        deep_supervision=False,
        c_list=None,
        bridge=True,
        frequency_mode="add",
        frequency_stages="none",
        frequency_gate=True,
        frequency_gamma=0.0,
        frequency_high_ratio=0.25,
        frequency_experts="add,dwt,gabor,dct",
        moe_top_k=2,
        moe_temperature=1.0,
        moe_prior="auto",
        moe_balance_weight=0.0,
        moe_entropy_weight=0.0,
        use_align=False,
        align_stages="none",
        align_max_flow=4.0,
        context_module="none",
        context_rates="1,2,3",
        context_dropout=0.0,
        related_models_root=None,
        **kwargs
    ):
        if not bridge:
            raise ValueError("EGE-UNet graft requires bridge=True in the bundled implementation.")
        c_list = c_list or [8, 16, 24, 32, 48, 64]
        super().__init__()
        self.related_models_root = str(_related_root(related_models_root))
        self.backbone = _load_ege_class(self.related_models_root)(
            num_classes=num_classes,
            input_channels=input_channels,
            c_list=c_list,
            bridge=bridge,
            gt_ds=True,
        )
        self._init_graft(
            {"t1": c_list[0], "t2": c_list[1], "t3": c_list[2], "t4": c_list[3], "t5": c_list[4]},
            c_list[5],
            frequency_mode,
            frequency_stages,
            frequency_gate,
            frequency_gamma,
            frequency_high_ratio,
            frequency_experts,
            moe_top_k,
            moe_temperature,
            moe_prior,
            moe_balance_weight,
            moe_entropy_weight,
            use_align,
            align_stages,
            align_max_flow,
            context_module,
            context_rates,
            context_dropout,
        )

    def forward(self, x):
        backbone = self.backbone
        out = F.gelu(F.max_pool2d(backbone.ebn1(backbone.encoder1(x)), 2, 2))
        t1 = out
        out = F.gelu(F.max_pool2d(backbone.ebn2(backbone.encoder2(out)), 2, 2))
        t2 = out
        out = F.gelu(F.max_pool2d(backbone.ebn3(backbone.encoder3(out)), 2, 2))
        t3 = out
        out = F.gelu(F.max_pool2d(backbone.ebn4(backbone.encoder4(out)), 2, 2))
        t4 = out
        out = F.gelu(F.max_pool2d(backbone.ebn5(backbone.encoder5(out)), 2, 2))
        t5 = out
        out = self._graft_context(F.gelu(backbone.encoder6(out)))

        out5 = F.gelu(backbone.dbn1(backbone.decoder1(out)))
        gt_pre5 = backbone.gt_conv1(out5)
        t5 = backbone.GAB5(out, t5, gt_pre5)
        out5 = self._graft_add("t5", out5, t5)
        out4 = F.gelu(F.interpolate(backbone.dbn2(backbone.decoder2(out5)), scale_factor=2, mode="bilinear", align_corners=False))
        gt_pre4 = backbone.gt_conv2(out4)
        t4 = backbone.GAB4(t5, t4, gt_pre4)
        out4 = self._graft_add("t4", out4, t4)
        out3 = F.gelu(F.interpolate(backbone.dbn3(backbone.decoder3(out4)), scale_factor=2, mode="bilinear", align_corners=False))
        gt_pre3 = backbone.gt_conv3(out3)
        t3 = backbone.GAB3(t4, t3, gt_pre3)
        out3 = self._graft_add("t3", out3, t3)
        out2 = F.gelu(F.interpolate(backbone.dbn4(backbone.decoder4(out3)), scale_factor=2, mode="bilinear", align_corners=False))
        gt_pre2 = backbone.gt_conv4(out2)
        t2 = backbone.GAB2(t3, t2, gt_pre2)
        out2 = self._graft_add("t2", out2, t2)
        out1 = F.gelu(F.interpolate(backbone.dbn5(backbone.decoder5(out2)), scale_factor=2, mode="bilinear", align_corners=False))
        gt_pre1 = backbone.gt_conv5(out1)
        t1 = backbone.GAB1(t2, t1, gt_pre1)
        out1 = self._graft_add("t1", out1, t1)
        return F.interpolate(backbone.final(out1), scale_factor=2, mode="bilinear", align_corners=False)


def build_external_model(arch, related_models_root=None, **kwargs):
    classes = {
        "UNet": UNet,
        "UNetPlusPlus": UNetPlusPlus,
        "UNet3Plus": UNet3Plus,
        "MALUNet": MALUNet,
        "EGEUNet": EGEUNet,
    }
    if arch not in classes:
        raise ValueError("Unsupported external architecture: {}".format(arch))
    kwargs["related_models_root"] = related_models_root
    return classes[arch](**kwargs)
