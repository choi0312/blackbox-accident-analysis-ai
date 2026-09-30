import torch
import torch.nn as nn
import torch.nn.functional as F


# This to alterative to Dynamic nn.AdaptiveAvgPool3d for onnx export
class AdaptiveAvgPool3dStatic(nn.Module):
    def __init__(self, mode="none_1_1"):
        super().__init__()
        # 'mode' can be 'none_1_1' to match (None, 1, 1) or '1' to match (1, 1, 1)
        self.mode = mode

    def forward(self, x):
        if self.mode == "1":
            # Apply adaptive pooling to all spatial dimensions to (1, 1, 1)
            x = x.mean(dim=(-3, -2, -1), keepdim=True)
        elif self.mode == "none_1_1":
            # Apply adaptive pooling to (None, 1, 1), preserve depth
            x = x.mean(dim=(-1, -2), keepdim=True)
        return x
    

class ContextualMotionAnalysis(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContextualMotionAnalysis, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels // 2, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(out_channels // 2, out_channels, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels // 2)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        motion = torch.abs(x[:, :, 1:] - x[:, :, :-1])
        motion = F.pad(motion, (0, 0, 0, 0, 1, 0), "constant", 0)
        motion = self.relu(self.bn1(self.conv1(motion)))
        motion = self.relu(self.bn2(self.conv2(motion)))
        return motion


class AdaptiveFeatureTransformer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveFeatureTransformer, self).__init__()
        self.attention_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.attention_bn = nn.BatchNorm3d(out_channels)
        self.attention_sigmoid = nn.Sigmoid()
        self.feature_transform = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1)
        self.feature_bn = nn.BatchNorm3d(out_channels)
        self.feature_relu = nn.ReLU(inplace=True)
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, motion):
        attention_weights = self.attention_sigmoid(self.attention_bn(self.attention_conv(motion)))
        transformed_features = self.feature_relu(self.feature_bn(self.feature_transform(x)))
        residual = self.residual(x)
        output = attention_weights * transformed_features + residual
        return output
    

class SpatialFeatureExtraction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialFeatureExtraction, self).__init__()
        
        # Depthwise separable convolution: Efficient spatial feature extraction
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        
        # Adjusted Dilated convolution: Captures broader context without reducing resolution
        self.dilated_conv = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), dilation=2, padding=(0, 2, 2))
        
        # Batch normalization and ReLU activation
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual connection
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.res_bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        residual = self.res_bn(self.residual(x))

        # Depthwise separable convolution
        x = self.depthwise(x)
        x = self.bn1(self.pointwise(x))
        x = self.relu(x)
        
        # Dilated convolution for broader context
        x = self.dilated_conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Combine with residual
        x = x + residual
        x = self.relu(x)
        
        return x
    

class MotionFeatureExtraction(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(MotionFeatureExtraction, self).__init__()
        
        # Initial 3D convolution to capture basic motion patterns across frames
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Temporal convolution layers to enhance motion feature extraction
        self.temporal_conv1 = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.temporal_conv2 = nn.Sequential(
            nn.Conv3d(mid_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Pooling layer to focus on motion features by reducing spatial dimensions
        self.temporal_pool = AdaptiveAvgPool3dStatic(mode="none_1_1")

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.temporal_conv1(x)
        x = self.temporal_conv2(x)
        x = self.temporal_pool(x)  # Reduce spatial dimensions to focus on motion features
        return x
    

class DynamicIntegrationGate(nn.Module):
    def __init__(self, output_channels):
        super(DynamicIntegrationGate, self).__init__()

        self.gate_network = nn.Sequential(
            nn.Conv3d(output_channels * 2, output_channels, kernel_size=1),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True),
            AdaptiveAvgPool3dStatic(mode="1"),
            nn.Conv3d(output_channels, output_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, spatial_features, temporal_features):
        target_size = spatial_features.size()[2:]  # Extracts the spatial dimensions (D, H, W)
        
        # Dynamically upsample the motion features to match the spatial features' size
        temporal_features_upsampled = F.interpolate(temporal_features, size=target_size, mode='trilinear', align_corners=False)
        
        combined_features = torch.cat([spatial_features, temporal_features_upsampled], dim=1)
        
        gating_coefficients = self.gate_network(combined_features)
        
        output = gating_coefficients * spatial_features + (1 - gating_coefficients) * temporal_features_upsampled
        return output



