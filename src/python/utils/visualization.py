"""
Visualization utilities for Project SENTINEL
Includes functions for point cloud visualization, label coloring, and result plotting
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from typing import Optional, List, Tuple, Union
import seaborn as sns

# Try importing Open3D (optional for better 3D visualization)
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not installed. Using matplotlib for visualization.")

def visualize_pointcloud(points: Union[np.ndarray, torch.Tensor], 
                        labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
                        predictions: Optional[Union[np.ndarray, torch.Tensor]] = None,
                        title: str = "Point Cloud Visualization",
                        save_path: Optional[str] = None,
                        max_points: int = 10000,
                        use_open3d: bool = True,
                        figsize: Tuple[int, int] = (15, 10),
                        point_size: float = 0.5) -> None:
    """
    Visualize point cloud with optional semantic labels or predictions
    
    Args:
        points: Point cloud coordinates [N, 3]
        labels: Ground truth labels [N]
        predictions: Predicted labels [N]
        title: Title for the visualization
        save_path: Path to save the visualization
        max_points: Maximum number of points to display
        use_open3d: Use Open3D if available (better for 3D)
        figsize: Figure size for matplotlib
        point_size: Size of points in visualization
    """
    # Convert torch tensors to numpy if necessary
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    
    # Ensure points is 2D array
    if len(points.shape) == 3:
        points = points.reshape(-1, 3)
    if labels is not None and len(labels.shape) > 1:
        labels = labels.flatten()
    if predictions is not None and len(predictions.shape) > 1:
        predictions = predictions.flatten()
    
    # Sample if too many points
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        if labels is not None:
            labels = labels[indices]
        if predictions is not None:
            predictions = predictions[indices]
    
    # Use Open3D if available and requested
    if HAS_OPEN3D and use_open3d:
        _visualize_open3d(points, labels, predictions, title, save_path)
    else:
        _visualize_matplotlib(points, labels, predictions, title, save_path, 
                            figsize, point_size)

def _visualize_open3d(points: np.ndarray,
                     labels: Optional[np.ndarray],
                     predictions: Optional[np.ndarray],
                     title: str,
                     save_path: Optional[str]) -> None:
    """
    Visualize using Open3D (better 3D interaction)
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Color by labels or predictions
    if predictions is not None and labels is not None:
        # Show both - split view
        colors_gt = label_to_color(labels)
        colors_pred = label_to_color(predictions)
        
        # Create two point clouds side by side
        pcd_gt = o3d.geometry.PointCloud()
        pcd_pred = o3d.geometry.PointCloud()
        
        # Shift prediction cloud to the right
        points_shifted = points.copy()
        points_shifted[:, 0] += np.max(points[:, 0]) - np.min(points[:, 0]) + 5
        
        pcd_gt.points = o3d.utility.Vector3dVector(points)
        pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt)
        
        pcd_pred.points = o3d.utility.Vector3dVector(points_shifted)
        pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred)
        
        # Add coordinate frames for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        
        # Visualize both
        o3d.visualization.draw_geometries(
            [pcd_gt, pcd_pred, coord_frame],
            window_name=f"{title} - Left: Ground Truth, Right: Predictions",
            width=1200,
            height=800
        )
        
    elif labels is not None:
        colors = label_to_color(labels)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], window_name=title)
        
    elif predictions is not None:
        colors = label_to_color(predictions)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], window_name=title)
    else:
        # No labels - use height-based coloring
        heights = points[:, 2]
        colors = height_to_color(heights)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], window_name=title)
    
    # Save if path provided
    if save_path:
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"Point cloud saved to {save_path}")

def _visualize_matplotlib(points: np.ndarray,
                         labels: Optional[np.ndarray],
                         predictions: Optional[np.ndarray],
                         title: str,
                         save_path: Optional[str],
                         figsize: Tuple[int, int],
                         point_size: float) -> None:
    """
    Visualize using matplotlib (fallback option)
    """
    if predictions is not None and labels is not None:
        # Create side-by-side comparison
        fig = plt.figure(figsize=figsize)
        
        # Ground truth
        ax1 = fig.add_subplot(121, projection='3d')
        colors_gt = label_to_color(labels)
        scatter1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                              c=colors_gt, s=point_size, alpha=0.6)
        ax1.set_title('Ground Truth')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.view_init(elev=20, azim=45)
        
        # Predictions
        ax2 = fig.add_subplot(122, projection='3d')
        colors_pred = label_to_color(predictions)
        scatter2 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
                              c=colors_pred, s=point_size, alpha=0.6)
        ax2.set_title('Predictions')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.view_init(elev=20, azim=45)
        
        plt.suptitle(title)
        
    else:
        # Single view
        fig = plt.figure(figsize=(figsize[0]//2, figsize[1]))
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None:
            colors = label_to_color(labels)
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                               c=colors, s=point_size, alpha=0.6)
        elif predictions is not None:
            colors = label_to_color(predictions)
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                               c=colors, s=point_size, alpha=0.6)
        else:
            # Color by height
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                               c=points[:, 2], cmap='viridis', 
                               s=point_size, alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Height (Z)')
        
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def label_to_color(labels: np.ndarray, num_classes: int = 20) -> np.ndarray:
    """
    Convert semantic labels to RGB colors
    
    Args:
        labels: Array of label indices
        num_classes: Total number of classes
    
    Returns:
        RGB colors array [N, 3] with values in [0, 1]
    """
    # Define color map for SemanticKITTI classes
    colors_map = np.array([
        [0, 0, 0],        # 0: unlabeled/unclassified - black
        [255, 0, 0],      # 1: car - red
        [0, 255, 0],      # 2: bicycle - green
        [0, 0, 255],      # 3: motorcycle - blue
        [255, 255, 0],    # 4: truck - yellow
        [255, 0, 255],    # 5: other-vehicle - magenta
        [0, 255, 255],    # 6: person - cyan
        [128, 0, 0],      # 7: bicyclist - dark red
        [0, 128, 0],      # 8: motorcyclist - dark green
        [128, 128, 128],  # 9: road - gray
        [64, 64, 64],     # 10: parking - dark gray
        [192, 192, 192],  # 11: sidewalk - light gray
        [128, 64, 0],     # 12: other-ground - brown
        [128, 0, 128],    # 13: building - purple
        [64, 64, 128],    # 14: fence - blue-gray
        [0, 128, 64],     # 15: vegetation - green
        [64, 128, 0],     # 16: trunk - olive
        [128, 128, 0],    # 17: terrain - olive
        [255, 128, 0],    # 18: pole - orange
        [255, 255, 128],  # 19: traffic-sign - light yellow
    ], dtype=np.float32) / 255.0
    
    # Extend color map if needed
    if num_classes > len(colors_map):
        # Generate additional colors
        np.random.seed(42)  # For reproducibility
        additional_colors = np.random.rand(num_classes - len(colors_map), 3)
        colors_map = np.vstack([colors_map, additional_colors])
    
    # Ensure labels are within valid range
    labels = np.clip(labels, 0, num_classes - 1)
    
    # Map labels to colors
    colors = colors_map[labels.astype(int)]
    
    return colors

def height_to_color(heights: np.ndarray) -> np.ndarray:
    """
    Convert height values to colors using a colormap
    
    Args:
        heights: Array of height values
    
    Returns:
        RGB colors array [N, 3]
    """
    # Normalize heights to [0, 1]
    h_min, h_max = heights.min(), heights.max()
    if h_max - h_min > 0:
        heights_norm = (heights - h_min) / (h_max - h_min)
    else:
        heights_norm = np.zeros_like(heights)
    
    # Use matplotlib colormap
    import matplotlib.cm as cm
    colormap = cm.get_cmap('viridis')
    colors = colormap(heights_norm)[:, :3]  # Remove alpha channel
    
    return colors

def plot_training_history(history: dict, 
                         save_path: Optional[str] = None,
                         show: bool = True) -> None:
    """
    Plot training history (loss, accuracy, mIoU, learning rate)
    
    Args:
        history: Dictionary containing training metrics
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss plot
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Loss over Epochs', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if 'train_acc' in history and 'val_acc' in history:
        axes[0, 1].plot(history['train_acc'], label='Train', linewidth=2)
        axes[0, 1].plot(history['val_acc'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Accuracy over Epochs', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # mIoU plot
    if 'train_miou' in history and 'val_miou' in history:
        axes[1, 0].plot(history['train_miou'], label='Train', linewidth=2)
        axes[1, 0].plot(history['val_miou'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Mean IoU over Epochs', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mIoU')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate plot
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], linewidth=2, color='orange')
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training History - Project SENTINEL', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         normalize: bool = True,
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 10),
                         show: bool = True) -> np.ndarray:
    """
    Plot confusion matrix for semantic segmentation
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        normalize: Whether to normalize the matrix
        save_path: Path to save the plot
        figsize: Figure size
        show: Whether to display the plot
    
    Returns:
        Confusion matrix array
    """
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Default class names if not provided
    if class_names is None:
        class_names = [
            'unclassified', 'car', 'bicycle', 'motorcycle', 'truck',
            'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road',
            'parking', 'sidewalk', 'other-ground', 'building', 'fence',
            'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
        ]
    
    # Use only the classes present in the data
    num_classes = cm.shape[0]
    if len(class_names) > num_classes:
        class_names = class_names[:num_classes]
    
    # Create plot
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Frequency' if normalize else 'Count'})
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''),
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return cm

def plot_per_class_metrics(metrics: dict,
                          class_names: Optional[List[str]] = None,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (14, 8),
                          show: bool = True) -> None:
    """
    Plot per-class metrics (IoU, Precision, Recall)
    
    Args:
        metrics: Dictionary containing per-class metrics
        class_names: Names of classes
        save_path: Path to save the plot
        figsize: Figure size
        show: Whether to display the plot
    """
    # Default class names
    if class_names is None:
        class_names = [
            'unclass.', 'car', 'bicycle', 'moto.', 'truck',
            'other-v.', 'person', 'bicycl.', 'motocyc.', 'road',
            'parking', 'sidewalk', 'other-g.', 'building', 'fence',
            'veget.', 'trunk', 'terrain', 'pole', 'sign'
        ]
    
    # Prepare data
    num_classes = len(metrics.get('per_class_iou', []))
    x_pos = np.arange(num_classes)
    
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # IoU plot
    if 'per_class_iou' in metrics:
        axes[0].bar(x_pos, metrics['per_class_iou'], color='steelblue', alpha=0.8)
        axes[0].set_ylabel('IoU')
        axes[0].set_title('Per-Class Intersection over Union', fontweight='bold')
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].axhline(y=np.mean(metrics['per_class_iou']), 
                       color='red', linestyle='--', 
                       label=f"Mean: {np.mean(metrics['per_class_iou']):.3f}")
        axes[0].legend()
    
    # Precision plot
    if 'per_class_precision' in metrics:
        axes[1].bar(x_pos, metrics['per_class_precision'], color='green', alpha=0.8)
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Per-Class Precision', fontweight='bold')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].axhline(y=np.mean(metrics['per_class_precision']), 
                       color='red', linestyle='--',
                       label=f"Mean: {np.mean(metrics['per_class_precision']):.3f}")
        axes[1].legend()
    
    # Recall plot
    if 'per_class_recall' in metrics:
        axes[2].bar(x_pos, metrics['per_class_recall'], color='orange', alpha=0.8)
        axes[2].set_ylabel('Recall')
        axes[2].set_title('Per-Class Recall', fontweight='bold')
        axes[2].set_ylim([0, 1])
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].axhline(y=np.mean(metrics['per_class_recall']), 
                       color='red', linestyle='--',
                       label=f"Mean: {np.mean(metrics['per_class_recall']):.3f}")
        axes[2].legend()
    
    # Set x-axis labels
    for ax in axes:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(class_names[:num_classes], rotation=45, ha='right')
    
    plt.suptitle('Per-Class Metrics Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-class metrics plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def create_comparison_figure(original_points: np.ndarray,
                            baseline_predictions: np.ndarray,
                            refined_predictions: np.ndarray,
                            ground_truth: Optional[np.ndarray] = None,
                            save_path: Optional[str] = None,
                            max_points: int = 5000,
                            show: bool = True) -> None:
    """
    Create a comparison figure showing baseline vs refined predictions
    
    Args:
        original_points: Point cloud coordinates
        baseline_predictions: Predictions from baseline model
        refined_predictions: Predictions after geometric refinement
        ground_truth: Optional ground truth labels
        save_path: Path to save the figure
        max_points: Maximum points to display
        show: Whether to display the plot
    """
    # Sample points if necessary
    if len(original_points) > max_points:
        indices = np.random.choice(len(original_points), max_points, replace=False)
        original_points = original_points[indices]
        baseline_predictions = baseline_predictions[indices]
        refined_predictions = refined_predictions[indices]
        if ground_truth is not None:
            ground_truth = ground_truth[indices]
    
    # Determine number of subplots
    num_plots = 3 if ground_truth is None else 4
    fig = plt.figure(figsize=(5 * num_plots, 5))
    
    plot_idx = 1
    
    # Ground truth (if available)
    if ground_truth is not None:
        ax = fig.add_subplot(1, num_plots, plot_idx, projection='3d')
        colors = label_to_color(ground_truth)
        ax.scatter(original_points[:, 0], original_points[:, 1], 
                  original_points[:, 2], c=colors, s=0.5, alpha=0.6)
        ax.set_title('Ground Truth')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plot_idx += 1
    
    # Baseline predictions
    ax = fig.add_subplot(1, num_plots, plot_idx, projection='3d')
    colors = label_to_color(baseline_predictions)
    ax.scatter(original_points[:, 0], original_points[:, 1], 
              original_points[:, 2], c=colors, s=0.5, alpha=0.6)
    ax.set_title('Baseline (PointNet++)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plot_idx += 1
    
    # Refined predictions
    ax = fig.add_subplot(1, num_plots, plot_idx, projection='3d')
    colors = label_to_color(refined_predictions)
    ax.scatter(original_points[:, 0], original_points[:, 1], 
              original_points[:, 2], c=colors, s=0.5, alpha=0.6)
    ax.set_title('SENTINEL (Refined)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plot_idx += 1
    
    # Difference map
    ax = fig.add_subplot(1, num_plots, plot_idx, projection='3d')
    differences = (baseline_predictions != refined_predictions).astype(float)
    colors = plt.cm.RdYlGn(1 - differences)[:, :3]  # Red for differences
    ax.scatter(original_points[:, 0], original_points[:, 1], 
              original_points[:, 2], c=colors, s=0.5, alpha=0.6)
    ax.set_title('Refinement Changes')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.suptitle('SENTINEL Geometric Refinement Comparison', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

# Export main functions
__all__ = [
    'visualize_pointcloud',
    'label_to_color',
    'height_to_color',
    'plot_training_history',
    'plot_confusion_matrix',
    'plot_per_class_metrics',
    'create_comparison_figure'
]