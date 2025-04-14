"""
File: spd/experiments/waterbird/models.py

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

class WaterbirdResNet18(nn.Module):
    """
    A ResNet18 model with two fully-connected layers at the end:
    - fc1 -> some hidden dimension
    - fc2 -> output dimension (2 for Bird classification, or  something else if needed)
    """

    def __init__(self, num_classes=2, hidden_dim=512):
        super().__init__()

        # Pretrained ResNet, remove final fc
        base_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base_resnet.children())[:-1])  # up to pool
        
        # We flatten after the global avgpool
        # Then define two FC layers
        in_features = base_resnet.fc.in_features  # typically 512 for resnet18
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # B, C, H, W
        feats = self.features(x)               # [B, 512, 1, 1] after the final conv
        feats = feats.flatten(start_dim=1)     # [B, 512]
        h = F.relu(self.fc1(feats))            # [B, hidden_dim]
        out = self.fc2(h)                      # [B, num_classes]
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


