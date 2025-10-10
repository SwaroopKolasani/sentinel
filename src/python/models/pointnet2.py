import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation

class PointNet2SemanticSegmentation(nn.Module):
    def __init__(self, num_classes=20):
        super(PointNet2SemanticSegmentation, self).__init__()
        
        # Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.1, nsample=32, 
            in_channel=6, mlp=[32, 32, 64], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.2, nsample=32,
            in_channel=64 + 3, mlp=[64, 64, 128], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.4, nsample=32,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=16, radius=0.8, nsample=32,
            in_channel=256 + 3, mlp=[256, 256, 512], group_all=False
        )
        
        # Feature Propagation layers
        self.fp4 = PointNetFeaturePropagation(
            in_channel=768, mlp=[256, 256]
        )
        self.fp3 = PointNetFeaturePropagation(
            in_channel=384, mlp=[256, 256]
        )
        self.fp2 = PointNetFeaturePropagation(
            in_channel=320, mlp=[256, 128]
        )
        self.fp1 = PointNetFeaturePropagation(
            in_channel=128, mlp=[128, 128, 128]
        )
        
        # Final classifier
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        
    def forward(self, xyz, features=None):
        B, N, _ = xyz.shape
        
        if features is None:
            features = xyz
        
        # Set Abstraction
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        
        # Feature Propagation
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)
        
        # Classification
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        
        return x