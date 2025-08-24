
from .advanced_swin_unet import AdvancedSwinUNet
from .unet import UNet
from .torchvision_wrap import get_torchvision_models
MODEL_REGISTRY = {"AdvancedSwinUNet": AdvancedSwinUNet, "UNet": UNet}
MODEL_REGISTRY.update(get_torchvision_models())
