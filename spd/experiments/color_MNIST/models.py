"""
Defines:
1) A ResNet18-based model with 2 FC layers at the end for Waterbirds classification.
2) An SPD-based final-layer version that wraps the same architecture but replaces
   the final FC layers with SPD decompositions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from spd.models.components import LinearComponent, TransposedLinearComponent
from spd.models.base import SPDModel
from spd.hooks import HookedRootModule
from spd.module_utils import init_param_

class TeacherFC(HookedRootModule):
    def __init__(self, fc1, fc2):
        super().__init__()
        self.fc1 = fc1
        self.fc2 = fc2
        self.setup()  # Important: This sets up the hook dict
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return out

class ColorMNISTConvNetGAP(nn.Module):
    """
    A lightweight convolutional network for colourised MNIST
    (input size 3 × 28 × 28) that uses *global average pooling* to
    keep the input to the fully-connected layers down to exactly
    28 × 28 = 784 features.

    Architecture:

        Conv(3 → 32, 3×3, padding=1) + ReLU
        Conv(32 → 64, 3×3, padding=1) + ReLU
        Channel-wise Global Average Pooling     # [B, 64, 28, 28] → [B, 28, 28]
        ──────────────────────────────────────────────────────────
        Flatten                                 # 784 features
        Linear(784 → hidden_dim) + ReLU
        Linear(hidden_dim → num_classes)
    """

    def __init__(self, num_classes: int = 10, hidden_dim: int = 128):
        super().__init__()

        # Convolutional feature extractor (spatial resolution stays 28×28)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # After channel-wise GAP we have 28×28 = 784 features
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, 3, 28, 28]
        returns: [batch, num_classes]
        """
        feats = self.conv(x)             # [B, 64, 28, 28]
        feats = feats.mean(dim=1)        # Channel-wise GAP → [B, 28, 28]
        feats = feats.flatten(start_dim=1)  # [B, 784]
        h = F.relu(self.fc1(feats))      # [B, hidden_dim]
        out = self.fc2(h)                # [B, num_classes]
        return out


#############################################
# SPD Variation Only On Final 2 FC Layers
#############################################

class SPDTwoLayerFC(SPDModel):
    """
    Replaces fc1 & fc2 with SPD decomposition. We keep the ResNet trunk fixed.
    The rest is SPD for the last two layers only.

    (We treat them as separate SPD layers, each with shape (C, d_in, m) & (C, m, d_out).)
    """

    def __init__(
        self,
        in_features: int = 512,     # from resnet18 conv trunk
        hidden_dim: int = 512,      # same hidden as above
        num_classes: int = 2,       # classify bird vs. not
        C: int = 40,                # number of subcomponents
        m_fc1: int = 16,            # rank dimension for first FC
        m_fc2: int = 16,            # rank dimension for second FC
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.C = C

        # SPD layer 1: from in_features -> hidden_dim
        self.fc1 = LinearComponent(
            d_in=in_features,
            d_out=hidden_dim,
            C=C,
            n_instances=None,
            init_type="xavier_normal",
            init_scale=1.0,
            m=m_fc1,
        )
        # SPD layer 2: from hidden_dim -> num_classes
        self.fc2 = LinearComponent(
            d_in=hidden_dim,
            d_out=num_classes,
            C=C,
            n_instances=None,
            init_type="xavier_normal",
            init_scale=1.0,
            m=m_fc2,
        )
        self.setup()  # from HookedRootModule

    def forward(self, x, topk_mask=None):
        """
        x: shape [B, hidden_dim_in], i.e. the output from the ResNet trunk
        topk_mask: optional boolean mask [B, C], if we want to ablate subnets.
        """
        # FC1
        h = self.fc1(x, topk_mask=topk_mask)
        h = F.relu(h)

        # FC2
        out = self.fc2(h, topk_mask=topk_mask)
        return out


