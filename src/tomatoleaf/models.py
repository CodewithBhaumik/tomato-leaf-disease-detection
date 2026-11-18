"""Deep learning architectures: a custom CNN trained from scratch and MobileNetV2 transfer learning."""
import torch
from torch import nn
from torchvision.models import (ConvNeXt_Tiny_Weights, EfficientNet_B0_Weights, MobileNet_V2_Weights, convnext_tiny,
                                efficientnet_b0, mobilenet_v2)

from .config import NUM_CLASSES


def conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CustomCNN(nn.Module):
    """4 blocks of [conv-BN-ReLU x2 -> max-pool], global average pooling, dropout, linear classifier.

    224x224 input -> 112 -> 56 -> 28 -> 14 spatial resolution; ~1.2M parameters.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, channels=(32, 64, 128, 256), dropout: float = 0.4):
        super().__init__()
        blocks, in_ch = [], 3
        for out_ch in channels:
            blocks.append(nn.Sequential(conv_bn_relu(in_ch, out_ch), conv_bn_relu(out_ch, out_ch), nn.MaxPool2d(2)))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def build_mobilenetv2(num_classes: int = NUM_CLASSES, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """ImageNet-pretrained MobileNetV2 with a new 8-class head (1280 -> 8)."""
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.last_channel, num_classes))
    return model


def build_efficientnet_b0(num_classes: int = NUM_CLASSES, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """ImageNet-pretrained EfficientNet-B0 (compound-scaled MBConv network, 5.3M parameters) with an 8-class head."""
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier[1].in_features, num_classes))
    return model


def build_convnext_tiny(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """ImageNet-pretrained ConvNeXt-Tiny (modernised ResNet: 7x7 depthwise convs, LayerNorm, GELU; 28M parameters)."""
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


def build_model(arch: str, pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    if arch == "custom_cnn":
        return CustomCNN(num_classes=num_classes)
    if arch == "mobilenetv2":
        return build_mobilenetv2(num_classes=num_classes, pretrained=pretrained)
    if arch == "efficientnet_b0":
        return build_efficientnet_b0(num_classes=num_classes, pretrained=pretrained)
    if arch == "convnext_tiny":
        return build_convnext_tiny(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown architecture: {arch}")


def load_checkpoint(path, device="cpu"):
    """Rebuild the architecture with as many outputs as the checkpoint's class list, load weights, eval mode."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["arch"], pretrained=False, num_classes=len(ckpt["class_keys"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = trainable


def gradcam_layer(model: nn.Module, arch: str) -> nn.Module:
    """Last convolutional stage (highest-level spatial feature maps)."""
    return model.features[-1]


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
