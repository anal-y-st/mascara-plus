
def get_torchvision_models():
    try:
        import torchvision.models.segmentation as seg
        import torch.nn as nn
    except Exception:
        return {}
    def deeplab(num_classes=1):
        m = seg.deeplabv3_resnet50(weights=None, aux_loss=None)
        try: m.classifier[-1] = nn.Conv2d(256, num_classes, 1)
        except Exception: pass
        return m
    def fcn(num_classes=1):
        m = seg.fcn_resnet50(weights=None, aux_loss=None)
        try: m.classifier[-1] = nn.Conv2d(512, num_classes, 1)
        except Exception: pass
        return m
    return {"DeepLabV3_ResNet50": deeplab, "FCN_ResNet50": fcn}
