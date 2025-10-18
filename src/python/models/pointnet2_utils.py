import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """
    Calculate Euclidean distance between each two points.
    src: [B, N, C]
    dst: [B, M, C]
    Output: [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: [B, N, C]
        idx: [B, S] or [B, S, K]
    Return:
        new_points: [B, S, C] or [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]

    if idx.dim() == 2:
        # Index shape: [B, S]
        batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1)
        return points[batch_indices, idx]  # [B, S, C]

    elif idx.dim() == 3:
        # Index shape: [B, S, K]
        B, S, K = idx.shape
        batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1, 1)
        batch_indices = batch_indices.repeat(1, S, K)  # [B, S, K]
        return points[batch_indices, idx]  # [B, S, K, C]

    else:
        raise ValueError(f"Unsupported idx shape: {idx.shape}")



def farthest_point_sample(xyz, npoint):
    """
    Sample farthest points.
    xyz: [B, N, 3]
    Return: [B, npoint]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Group points within a radius around new_xyz.
    xyz: [B, N, 3]
    new_xyz: [B, S, 3]
    Return: group_idx: [B, S, nsample]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    
    sqrdists = square_distance(new_xyz, xyz)
    
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    group_idx[sqrdists > radius ** 2] = N
    
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Sample and group for set abstraction
    """
    B, N, C = xyz.shape
    S = npoint
    
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, 3)
    
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Sample all points as a single group (for global features)
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    
    return new_xyz, new_points