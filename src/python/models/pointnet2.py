import torch
import torch.nn as nn
import torch.nn.functional as F
from src.python.models.pointnet2_utils import sample_and_group, sample_and_group_all, square_distance, index_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: [B, N, 3]
            points: [B, N, D]
        Return:
            new_xyz: [B, S, 3]
            new_points: [B, S, D']
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        
        # new_points: [B, npoint, nsample, channel]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, channel, nsample, npoint]
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        new_points = torch.max(new_points, 2)[0]
        new_points = new_points.permute(0, 2, 1)
        
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: [B, N, 3] - points to interpolate to
            xyz2: [B, S, 3] - points to interpolate from
            points1: [B, N, D] - features for xyz1 (can be None)
            points2: [B, S, D] - features for xyz2
        Return:
            new_points: [B, N, D']
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape
        
        if S == 1:
            # If only one point in xyz2, repeat for all points in xyz1
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # Find 3 nearest neighbors in xyz2 for each point in xyz1
            dists = square_distance(xyz1, xyz2)  # [B, N, S]
            dists, idx = dists.sort(dim=-1)      # Sort by distance
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # Take 3 nearest
            
            # Inverse distance weighting
            weights = 1.0 / (dists + 1e-8)
            weights = weights / weights.sum(dim=-1, keepdim=True)  # [B, N, 3]
            
            # Weighted interpolation
            interpolated_points = torch.zeros(B, N, points2.size(-1)).to(xyz1.device)
            for i in range(3):
                interpolated_points += index_points(points2, idx[:, :, i]) * weights[:, :, i:i+1]
        
        # Concatenate with points1 if available
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        
        # Apply MLPs
        new_points = new_points.permute(0, 2, 1)  # [B, D, N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        return new_points.permute(0, 2, 1)  # [B, N, D']


class PointNet2SemanticSegmentation(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        
        # Set Abstraction layers (encoder)
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 1, [32, 32, 64])
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64, [64, 64, 128])
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128, [128, 128, 256])
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256, [256, 256, 512])
        
        # Feature Propagation layers (decoder)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])  # 512 + 256
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])  # 256 + 128
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])  # 256 + 64
        self.fp1 = PointNetFeaturePropagation(129, [128, 128, 128])  # 128 + 1
        
        # Final classifier
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, points):
        """
        Input:
            points: [B, N, 4] - (x, y, z, intensity)
        Return:
            scores: [B, N, num_classes] - per-point class scores
        """
        B, N, _ = points.shape
        
        # Extract xyz and features
        xyz = points[:, :, :3]
        features = points[:, :, 3:4] if points.shape[-1] > 3 else None
        
        # Set Abstraction (encoding)
        l0_xyz = xyz
        l0_points = features
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        
        # Feature Propagation (decoding)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        
        # Final classification
        feat = l0_points.permute(0, 2, 1)  # [B, D, N]
        feat = F.relu(self.bn1(self.conv1(feat)))
        feat = self.drop1(feat)
        x = self.conv2(feat)  # [B, num_classes, N]
        x = x.permute(0, 2, 1)  # [B, N, num_classes]
        
        return x