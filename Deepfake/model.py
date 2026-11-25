import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class DeepfakeDetector(nn.Module):
    """
    EfficientNet-B0 fine-tuned for deepfake detection.
    Pretrained on ImageNet — generalises to both face-swap deepfakes
    and AI-generated / diffusion-model synthetic faces.
    """

    def __init__(self, num_classes=2):
        super(DeepfakeDetector, self).__init__()
        # Load EfficientNet-B0 with pretrained ImageNet weights
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Replace the final classifier head for binary (real / fake) classification
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Legacy alias so that existing code which does `from model import SimpleCNN`
# still works without modification.
# ---------------------------------------------------------------------------
SimpleCNN = DeepfakeDetector
