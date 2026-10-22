import archs_GBC
import external_models


EXTERNAL_ARCH_NAMES = tuple(external_models.EXTERNAL_ARCH_NAMES)
ARCH_NAMES = tuple(archs_GBC.__all__) + EXTERNAL_ARCH_NAMES


def build_model(
    arch,
    related_models_root=None,
    base_channels=32,
    **kwargs
):
    if arch in EXTERNAL_ARCH_NAMES:
        if arch == "UNetPlusPlus":
            kwargs["base_channels"] = base_channels
        return external_models.build_external_model(
            arch,
            related_models_root=related_models_root,
            **kwargs
        )

    if arch in archs_GBC.__all__:
        return archs_GBC.__dict__[arch](**kwargs)

    choices = ", ".join(ARCH_NAMES)
    raise ValueError("Unsupported architecture '{}'. Choices: {}".format(arch, choices))
