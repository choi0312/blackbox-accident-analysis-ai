import torch
import torch.nn as nn
from model.modules import (
    ContextualMotionAnalysis,
    AdaptiveFeatureTransformer,
    SpatialFeatureExtraction,
    MotionFeatureExtraction,
    DynamicIntegrationGate,
    AdaptiveAvgPool3dStatic
)

class FlexiNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=1):
        super(FlexiNet, self).__init__()
        self.cma_block = ContextualMotionAnalysis(input_channels, 64)
        self.aft_block = AdaptiveFeatureTransformer(64, 128)
        self.sfe_module = SpatialFeatureExtraction(128, 256)
        self.mfe_module = MotionFeatureExtraction(128, 192, 256)
        self.dig = DynamicIntegrationGate(output_channels=256)

        self.global_pool = AdaptiveAvgPool3dStatic(mode="1")
        self.fc = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = x.permute(0,2,1,3,4)  # Permute to match expected dimensions
        cma_out = self.cma_block(x)
        aft_out = self.aft_block(cma_out, cma_out)
        sfe_out = self.sfe_module(aft_out)
        mfe_out = self.mfe_module(aft_out)
        integrated = self.dig(sfe_out, mfe_out)
        
        pooled = self.global_pool(integrated).view(integrated.size(0), -1)
        out = self.fc(pooled)
        return out
