"""
Data augmentation utilities for Project SENTINEL
Includes various point cloud augmentation techniques for training robust models
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional, Union, List
import random


class PointCloudAugmentation:
    """
    Point cloud augmentation class with various transformation methods
    for improving model robustness and generalization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize augmentation with configuration
        
        Args:
            config: Dictionary containing augmentation parameters
        """
        self.config = config or {}
        
        # Default parameters if not specified
        self.defaults = {
            # Rotation
            'random_rotate': True,
            'rotate_range': (-np.pi, np.pi),  # Full rotation for Z-axis
            'rotate_axis': 'z',
            
            # Scaling
            'random_scale': True,
            'scale_range': (0.9, 1.1),
            
            # Translation
            'random_translate': False,
            'translate_range': (-1.0, 1.0),
            
            # Jittering (noise)
            'random_jitter': True,
            'jitter_std': 0.01,
            'jitter_clip': 0.05,
            
            # Dropout
            'random_dropout': True,
            'dropout_ratio': (0.0, 0.2),
            
            # Flip
            'random_flip': False,
            'flip_axis': [0, 1],  # X and Y axis
            
            # Elastic distortion
            'elastic_distortion': False,
            'elastic_alpha': 0.5,
            'elastic_sigma': 0.1,
            
            # Shear
            'random_shear': False,
            'shear_range': (-0.2, 0.2),
            
            # Mix-up
            'mixup': False,
            'mixup_alpha': 0.2,
            
            # Cutmix
            'cutmix': False,
            'cutmix_prob': 0.5,
            
            # Global dropout (remove entire objects)
            'global_dropout': False,
            'global_dropout_ratio': 0.1,
        }
        
        # Update defaults with provided config
        for key, value in self.config.items():
            self.defaults[key] = value
        
        # Set random seed if provided
        if 'seed' in self.config:
            np.random.seed(self.config['seed'])
            torch.manual_seed(self.config['seed'])
    
    def __call__(self, 
                 points: Union[np.ndarray, torch.Tensor],
                 labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 features: Optional[Union[np.ndarray, torch.Tensor]] = None
                 ) -> Tuple:
        """
        Apply augmentations to point cloud
        
        Args:
            points: Point cloud coordinates [N, 3] or [B, N, 3]
            labels: Optional semantic labels [N] or [B, N]
            features: Optional additional features [N, F] or [B, N, F]
        
        Returns:
            Augmented (points, labels, features) tuple
        """
        # Convert to numpy for processing
        is_tensor = isinstance(points, torch.Tensor)
        if is_tensor:
            device = points.device
            points = points.cpu().numpy()
            if labels is not None:
                labels = labels.cpu().numpy()
            if features is not None:
                features = features.cpu().numpy()
        
        # Apply augmentations based on configuration
        if self._get_param('random_rotate'):
            points = self.random_rotate(points)
        
        if self._get_param('random_scale'):
            points = self.random_scale(points)
        
        if self._get_param('random_translate'):
            points = self.random_translate(points)
        
        if self._get_param('random_flip'):
            points = self.random_flip(points)
        
        if self._get_param('random_jitter'):
            points = self.random_jitter(points)
        
        if self._get_param('random_shear'):
            points = self.random_shear(points)
        
        if self._get_param('elastic_distortion'):
            points = self.elastic_distortion(points)
        
        if self._get_param('random_dropout'):
            points, labels, features = self.random_dropout(points, labels, features)
        
        if self._get_param('global_dropout') and labels is not None:
            points, labels, features = self.global_dropout(points, labels, features)
        
        # Convert back to tensor if input was tensor
        if is_tensor:
            points = torch.from_numpy(points).float().to(device)
            if labels is not None:
                labels = torch.from_numpy(labels).long().to(device)
            if features is not None:
                features = torch.from_numpy(features).float().to(device)
        
        return points, labels, features
    
    def _get_param(self, key: str):
        """Get parameter value from config or defaults"""
        return self.defaults.get(key, False)
    
    def random_rotate(self, points: np.ndarray) -> np.ndarray:
        """
        Random rotation augmentation
        
        Args:
            points: Input points [N, 3] or [B, N, 3]
        
        Returns:
            Rotated points
        """
        rotate_range = self._get_param('rotate_range')
        rotate_axis = self._get_param('rotate_axis')
        
        # Random rotation angle
        angle = np.random.uniform(rotate_range[0], rotate_range[1])
        
        # Create rotation matrix based on axis
        if rotate_axis == 'x':
            rotation_matrix = self._rotation_matrix_x(angle)
        elif rotate_axis == 'y':
            rotation_matrix = self._rotation_matrix_y(angle)
        elif rotate_axis == 'z':
            rotation_matrix = self._rotation_matrix_z(angle)
        elif rotate_axis == 'all':
            # Random rotation on all axes
            angle_x = np.random.uniform(rotate_range[0], rotate_range[1])
            angle_y = np.random.uniform(rotate_range[0], rotate_range[1])
            angle_z = np.random.uniform(rotate_range[0], rotate_range[1])
            rotation_matrix = (self._rotation_matrix_x(angle_x) @
                             self._rotation_matrix_y(angle_y) @
                             self._rotation_matrix_z(angle_z))
        else:
            rotation_matrix = self._rotation_matrix_z(angle)
        
        # Apply rotation
        if len(points.shape) == 2:
            # Single point cloud
            points_rotated = points @ rotation_matrix.T
        else:
            # Batch of point clouds
            points_rotated = np.matmul(points, rotation_matrix.T)
        
        return points_rotated.astype(np.float32)
    
    def random_scale(self, points: np.ndarray) -> np.ndarray:
        """
        Random scaling augmentation
        
        Args:
            points: Input points
        
        Returns:
            Scaled points
        """
        scale_range = self._get_param('scale_range')
        
        # Uniform or per-axis scaling
        if np.random.random() < 0.5:
            # Uniform scaling
            scale = np.random.uniform(scale_range[0], scale_range[1])
            points_scaled = points * scale
        else:
            # Per-axis scaling
            scale_x = np.random.uniform(scale_range[0], scale_range[1])
            scale_y = np.random.uniform(scale_range[0], scale_range[1])
            scale_z = np.random.uniform(scale_range[0], scale_range[1])
            
            if len(points.shape) == 2:
                points_scaled = points * np.array([scale_x, scale_y, scale_z])
            else:
                points_scaled = points * np.array([scale_x, scale_y, scale_z])[None, None, :]
        
        return points_scaled.astype(np.float32)
    
    def random_translate(self, points: np.ndarray) -> np.ndarray:
        """
        Random translation augmentation
        
        Args:
            points: Input points
        
        Returns:
            Translated points
        """
        translate_range = self._get_param('translate_range')
        
        # Random translation vector
        translation = np.random.uniform(
            translate_range[0], 
            translate_range[1], 
            size=3
        )
        
        # Apply translation
        if len(points.shape) == 2:
            points_translated = points + translation
        else:
            points_translated = points + translation[None, None, :]
        
        return points_translated.astype(np.float32)
    
    def random_flip(self, points: np.ndarray) -> np.ndarray:
        """
        Random flip augmentation
        
        Args:
            points: Input points
        
        Returns:
            Flipped points
        """
        flip_axis = self._get_param('flip_axis')
        
        points_flipped = points.copy()
        
        # Flip along specified axes
        for axis in flip_axis:
            if np.random.random() < 0.5:
                if len(points.shape) == 2:
                    points_flipped[:, axis] = -points_flipped[:, axis]
                else:
                    points_flipped[:, :, axis] = -points_flipped[:, :, axis]
        
        return points_flipped.astype(np.float32)
    
    def random_jitter(self, points: np.ndarray) -> np.ndarray:
        """
        Add random Gaussian noise to points
        
        Args:
            points: Input points
        
        Returns:
            Jittered points
        """
        jitter_std = self._get_param('jitter_std')
        jitter_clip = self._get_param('jitter_clip')
        
        # Generate noise
        noise = np.random.normal(0, jitter_std, size=points.shape)
        noise = np.clip(noise, -jitter_clip, jitter_clip)
        
        # Add noise to points
        points_jittered = points + noise
        
        return points_jittered.astype(np.float32)
    
    def random_shear(self, points: np.ndarray) -> np.ndarray:
        """
        Random shear transformation
        
        Args:
            points: Input points
        
        Returns:
            Sheared points
        """
        shear_range = self._get_param('shear_range')
        
        # Random shear parameters
        shear_xy = np.random.uniform(shear_range[0], shear_range[1])
        shear_xz = np.random.uniform(shear_range[0], shear_range[1])
        shear_yx = np.random.uniform(shear_range[0], shear_range[1])
        shear_yz = np.random.uniform(shear_range[0], shear_range[1])
        shear_zx = np.random.uniform(shear_range[0], shear_range[1])
        shear_zy = np.random.uniform(shear_range[0], shear_range[1])
        
        # Shear matrix
        shear_matrix = np.array([
            [1, shear_xy, shear_xz],
            [shear_yx, 1, shear_yz],
            [shear_zx, shear_zy, 1]
        ])
        
        # Apply shear
        if len(points.shape) == 2:
            points_sheared = points @ shear_matrix.T
        else:
            points_sheared = np.matmul(points, shear_matrix.T)
        
        return points_sheared.astype(np.float32)
    
    def elastic_distortion(self, points: np.ndarray) -> np.ndarray:
        """
        Elastic distortion augmentation
        
        Args:
            points: Input points
        
        Returns:
            Distorted points
        """
        alpha = self._get_param('elastic_alpha')
        sigma = self._get_param('elastic_sigma')
        
        # Generate random displacement field
        if len(points.shape) == 2:
            num_points = points.shape[0]
            displacement = np.random.randn(num_points, 3) * alpha
            
            # Smooth displacement field (simplified version)
            for i in range(3):
                displacement[:, i] = np.convolve(
                    displacement[:, i], 
                    np.ones(5)/5, 
                    mode='same'
                )
        else:
            batch_size, num_points = points.shape[:2]
            displacement = np.random.randn(batch_size, num_points, 3) * alpha
        
        # Apply elastic distortion
        points_distorted = points + displacement * sigma
        
        return points_distorted.astype(np.float32)
    
    def random_dropout(self, 
                      points: np.ndarray,
                      labels: Optional[np.ndarray] = None,
                      features: Optional[np.ndarray] = None
                      ) -> Tuple:
        """
        Randomly drop points from the point cloud
        
        Args:
            points: Input points
            labels: Optional labels
            features: Optional features
        
        Returns:
            Tuple of (points, labels, features) after dropout
        """
        dropout_ratio = self._get_param('dropout_ratio')
        
        # Determine dropout ratio
        if isinstance(dropout_ratio, tuple):
            ratio = np.random.uniform(dropout_ratio[0], dropout_ratio[1])
        else:
            ratio = dropout_ratio
        
        if len(points.shape) == 2:
            # Single point cloud
            num_points = points.shape[0]
            keep_ratio = 1.0 - ratio
            num_keep = max(1, int(num_points * keep_ratio))
            
            # Random selection
            indices = np.random.choice(num_points, num_keep, replace=False)
            indices = np.sort(indices)  # Sort for consistency
            
            points_dropped = points[indices]
            labels_dropped = labels[indices] if labels is not None else None
            features_dropped = features[indices] if features is not None else None
            
        else:
            # Batch processing
            batch_size, num_points = points.shape[:2]
            keep_ratio = 1.0 - ratio
            num_keep = max(1, int(num_points * keep_ratio))
            
            points_list = []
            labels_list = []
            features_list = []
            
            for b in range(batch_size):
                indices = np.random.choice(num_points, num_keep, replace=False)
                indices = np.sort(indices)
                
                points_list.append(points[b, indices])
                if labels is not None:
                    labels_list.append(labels[b, indices])
                if features is not None:
                    features_list.append(features[b, indices])
            
            points_dropped = np.stack(points_list)
            labels_dropped = np.stack(labels_list) if labels is not None else None
            features_dropped = np.stack(features_list) if features is not None else None
        
        return points_dropped, labels_dropped, features_dropped
    
    def global_dropout(self,
                      points: np.ndarray,
                      labels: np.ndarray,
                      features: Optional[np.ndarray] = None
                      ) -> Tuple:
        """
        Drop entire objects/segments from the point cloud
        
        Args:
            points: Input points
            labels: Semantic labels
            features: Optional features
        
        Returns:
            Tuple of (points, labels, features) after global dropout
        """
        global_dropout_ratio = self._get_param('global_dropout_ratio')
        
        # Get unique labels (excluding background/unlabeled)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background
        
        if len(unique_labels) == 0:
            return points, labels, features
        
        # Randomly select labels to drop
        num_drop = int(len(unique_labels) * global_dropout_ratio)
        if num_drop > 0:
            labels_to_drop = np.random.choice(
                unique_labels, 
                min(num_drop, len(unique_labels)-1), 
                replace=False
            )
            
            # Create mask for points to keep
            keep_mask = np.ones(len(labels), dtype=bool)
            for label_to_drop in labels_to_drop:
                keep_mask &= (labels != label_to_drop)
            
            # Apply mask
            if len(points.shape) == 2:
                points_kept = points[keep_mask]
                labels_kept = labels[keep_mask]
                features_kept = features[keep_mask] if features is not None else None
            else:
                # For batch processing
                points_kept = points[:, keep_mask]
                labels_kept = labels[:, keep_mask]
                features_kept = features[:, keep_mask] if features is not None else None
            
            return points_kept, labels_kept, features_kept
        
        return points, labels, features
    
    def mixup(self,
             points1: np.ndarray,
             points2: np.ndarray,
             labels1: Optional[np.ndarray] = None,
             labels2: Optional[np.ndarray] = None,
             alpha: float = 0.2
             ) -> Tuple:
        """
        Mixup augmentation between two point clouds
        
        Args:
            points1: First point cloud
            points2: Second point cloud
            labels1: Labels for first point cloud
            labels2: Labels for second point cloud
            alpha: Mixup coefficient
        
        Returns:
            Mixed (points, labels) tuple
        """
        # Sample mixing coefficient from Beta distribution
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        
        # Mix points
        mixed_points = lam * points1 + (1 - lam) * points2
        
        # Mix labels (if provided)
        if labels1 is not None and labels2 is not None:
            # For classification, we typically keep the label with higher weight
            mixed_labels = labels1 if lam > 0.5 else labels2
        else:
            mixed_labels = None
        
        return mixed_points, mixed_labels, lam
    
    def cutmix(self,
              points1: np.ndarray,
              points2: np.ndarray,
              labels1: Optional[np.ndarray] = None,
              labels2: Optional[np.ndarray] = None,
              beta: float = 1.0
              ) -> Tuple:
        """
        CutMix augmentation for point clouds
        
        Args:
            points1: First point cloud
            points2: Second point cloud
            labels1: Labels for first point cloud
            labels2: Labels for second point cloud
            beta: Beta distribution parameter
        
        Returns:
            CutMixed (points, labels) tuple
        """
        # Sample mixing ratio
        lam = np.random.beta(beta, beta)
        
        num_points1 = len(points1)
        num_points2 = len(points2)
        
        # Determine number of points to take from each cloud
        num_from_1 = int(num_points1 * lam)
        num_from_2 = min(num_points2, num_points1 - num_from_1)
        
        # Random selection
        indices1 = np.random.choice(num_points1, num_from_1, replace=False)
        indices2 = np.random.choice(num_points2, num_from_2, replace=False)
        
        # Combine points
        mixed_points = np.vstack([
            points1[indices1],
            points2[indices2]
        ])
        
        # Combine labels if provided
        if labels1 is not None and labels2 is not None:
            mixed_labels = np.hstack([
                labels1[indices1],
                labels2[indices2]
            ])
        else:
            mixed_labels = None
        
        return mixed_points, mixed_labels
    
    @staticmethod
    def _rotation_matrix_x(angle: float) -> np.ndarray:
        """Rotation matrix around X axis"""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
    
    @staticmethod
    def _rotation_matrix_y(angle: float) -> np.ndarray:
        """Rotation matrix around Y axis"""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
    
    @staticmethod
    def _rotation_matrix_z(angle: float) -> np.ndarray:
        """Rotation matrix around Z axis"""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])


class AdverseWeatherAugmentation:
    """
    Specialized augmentations to simulate adverse weather conditions
    """
    
    def __init__(self):
        pass
    
    def add_rain_noise(self, 
                       points: np.ndarray,
                       intensity: float = 0.5,
                       drop_ratio: float = 0.1) -> np.ndarray:
        """
        Simulate rain effects on LiDAR
        
        Args:
            points: Input point cloud
            intensity: Rain intensity (0-1)
            drop_ratio: Ratio of points affected by rain drops
        
        Returns:
            Points with rain effects
        """
        num_points = len(points)
        
        # Add noise to simulate rain interference
        noise = np.random.randn(num_points, 3) * (intensity * 0.05)
        points_noisy = points + noise
        
        # Add random "rain drop" points in the air
        num_rain_points = int(num_points * drop_ratio * intensity)
        if num_rain_points > 0:
            # Generate rain points with bias towards being above the scene
            rain_points = np.random.randn(num_rain_points, 3)
            rain_points[:, 2] = np.abs(rain_points[:, 2]) * 2 + np.mean(points[:, 2])
            
            # Add some rain points
            points_with_rain = np.vstack([points_noisy, rain_points])
        else:
            points_with_rain = points_noisy
        
        return points_with_rain.astype(np.float32)
    
    def add_fog_occlusion(self,
                         points: np.ndarray,
                         density: float = 0.3,
                         visibility_range: float = 50.0) -> np.ndarray:
        """
        Simulate fog by reducing point density with distance
        
        Args:
            points: Input point cloud
            density: Fog density (0-1)
            visibility_range: Maximum visibility range in fog
        
        Returns:
            Points with fog occlusion
        """
        # Calculate distance from sensor (assumed at origin)
        distances = np.linalg.norm(points, axis=1)
        
        # Probability of keeping point decreases with distance
        # Using exponential decay model
        keep_prob = np.exp(-distances * density / visibility_range)
        keep_mask = np.random.random(len(points)) < keep_prob
        
        fog_points = points[keep_mask]
        
        # Add slight noise to simulate scattering
        if len(fog_points) > 0:
            noise = np.random.randn(*fog_points.shape) * (density * 0.02)
            fog_points = fog_points + noise
        
        return fog_points.astype(np.float32)
    
    def add_snow_effects(self,
                         points: np.ndarray,
                         accumulation: float = 0.3,
                         flake_ratio: float = 0.05) -> np.ndarray:
        """
        Simulate snow effects (accumulation and falling snow)
        
        Args:
            points: Input point cloud
            accumulation: Snow accumulation level (0-1)
            flake_ratio: Ratio of snowflake points
        
        Returns:
            Points with snow effects
        """
        # Raise ground level slightly (snow accumulation)
        points_snow = points.copy()
        ground_mask = points[:, 2] < np.percentile(points[:, 2], 20)
        points_snow[ground_mask, 2] += accumulation * 0.1
        
        # Add noise to simulate snow interference
        noise = np.random.randn(*points.shape) * (accumulation * 0.03)
        points_snow = points_snow + noise
        
        # Add falling snowflakes
        num_flakes = int(len(points) * flake_ratio * accumulation)
        if num_flakes > 0:
            snowflakes = np.random.randn(num_flakes, 3)
            snowflakes[:, 2] = np.random.uniform(
                np.min(points[:, 2]),
                np.max(points[:, 2]) + 2,
                num_flakes
            )
            points_snow = np.vstack([points_snow, snowflakes])
        
        # Randomly drop some points (occlusion by snow)
        keep_ratio = 1.0 - (accumulation * 0.2)
        num_keep = int(len(points_snow) * keep_ratio)
        if num_keep < len(points_snow):
            indices = np.random.choice(len(points_snow), num_keep, replace=False)
            points_snow = points_snow[indices]
        
        return points_snow.astype(np.float32)
    
    def add_dust_storm(self,
                       points: np.ndarray,
                       intensity: float = 0.4,
                       particle_ratio: float = 0.15) -> np.ndarray:
        """
        Simulate dust storm effects
        
        Args:
            points: Input point cloud
            intensity: Dust storm intensity (0-1)
            particle_ratio: Ratio of dust particles
        
        Returns:
            Points with dust storm effects
        """
        # Reduce overall visibility
        distances = np.linalg.norm(points[:, :2], axis=1)  # Distance in XY plane
        visibility_factor = np.exp(-distances * intensity * 0.05)
        keep_mask = np.random.random(len(points)) < visibility_factor
        
        dust_points = points[keep_mask]
        
        # Add dust particles
        num_particles = int(len(points) * particle_ratio * intensity)
        if num_particles > 0:
            # Dust particles distributed throughout the scene
            particles = np.random.randn(num_particles, 3)
            particles[:, :2] *= np.std(points[:, :2], axis=0) * 2
            particles[:, 2] = np.random.uniform(
                np.min(points[:, 2]),
                np.max(points[:, 2]) + 1,
                num_particles
            )
            dust_points = np.vstack([dust_points, particles])
        
        # Add heavy noise
        noise = np.random.randn(*dust_points.shape) * (intensity * 0.08)
        dust_points = dust_points + noise
        
        return dust_points.astype(np.float32)


# Functional API for convenience
def augment_point_cloud(points: Union[np.ndarray, torch.Tensor],
                       labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
                       config: Optional[Dict] = None) -> Tuple:
    """
    Functional interface for point cloud augmentation
    
    Args:
        points: Point cloud to augment
        labels: Optional labels
        config: Augmentation configuration
    
    Returns:
        Augmented (points, labels) tuple
    """
    augmenter = PointCloudAugmentation(config)
    return augmenter(points, labels)[:2]  # Return only points and labels


# Export main classes
__all__ = [
    'PointCloudAugmentation',
    'AdverseWeatherAugmentation',
    'augment_point_cloud'
]