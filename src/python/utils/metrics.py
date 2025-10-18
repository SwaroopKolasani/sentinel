"""
Evaluation metrics for Project SENTINEL
Includes IoU, accuracy, precision, recall, FPR, and other semantic segmentation metrics
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
import warnings


def calculate_iou(pred: Union[torch.Tensor, np.ndarray],
                  target: Union[torch.Tensor, np.ndarray],
                  num_classes: int,
                  ignore_index: int = -1) -> Tuple[float, List[float]]:
    """
    Calculate mean IoU and per-class IoU
    
    Args:
        pred: Predicted labels [N] or [B, N]
        target: Ground truth labels [N] or [B, N]
        num_classes: Number of classes
        ignore_index: Label index to ignore (e.g., unlabeled points)
    
    Returns:
        Tuple of (mean_iou, per_class_iou)
    """
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Flatten arrays
    pred = pred.flatten()
    target = target.flatten()
    
    # Remove ignored indices
    if ignore_index >= 0:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]
    
    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        # Calculate intersection and union
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        
        if union == 0:
            # If there are no ground truth and no predicted points for this class
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(union))
    
    # Calculate mean IoU, ignoring nan values
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0
    
    return mean_iou, ious


def calculate_accuracy(pred: Union[torch.Tensor, np.ndarray],
                      target: Union[torch.Tensor, np.ndarray],
                      ignore_index: int = -1) -> float:
    """
    Calculate overall accuracy
    
    Args:
        pred: Predicted labels
        target: Ground truth labels
        ignore_index: Label index to ignore
    
    Returns:
        Accuracy score
    """
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Flatten arrays
    pred = pred.flatten()
    target = target.flatten()
    
    # Remove ignored indices
    if ignore_index >= 0:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]
    
    if len(target) == 0:
        return 0.0
    
    correct = (pred == target).sum()
    total = len(target)
    
    return float(correct) / float(total)


def calculate_precision_recall(pred: Union[torch.Tensor, np.ndarray],
                              target: Union[torch.Tensor, np.ndarray],
                              num_classes: int,
                              ignore_index: int = -1) -> Tuple[float, List[float], float, List[float]]:
    """
    Calculate precision and recall for each class
    
    Args:
        pred: Predicted labels
        target: Ground truth labels
        num_classes: Number of classes
        ignore_index: Label index to ignore
    
    Returns:
        Tuple of (mean_precision, per_class_precision, mean_recall, per_class_recall)
    """
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Flatten arrays
    pred = pred.flatten()
    target = target.flatten()
    
    # Remove ignored indices
    if ignore_index >= 0:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]
    
    precisions = []
    recalls = []
    
    for cls in range(num_classes):
        # True Positives, False Positives, False Negatives
        tp = ((pred == cls) & (target == cls)).sum()
        fp = ((pred == cls) & (target != cls)).sum()
        fn = ((pred != cls) & (target == cls)).sum()
        
        # Precision: TP / (TP + FP)
        if tp + fp == 0:
            precisions.append(float('nan'))
        else:
            precisions.append(float(tp) / float(tp + fp))
        
        # Recall: TP / (TP + FN)
        if tp + fn == 0:
            recalls.append(float('nan'))
        else:
            recalls.append(float(tp) / float(tp + fn))
    
    # Calculate means, ignoring nan values
    valid_precisions = [p for p in precisions if not np.isnan(p)]
    valid_recalls = [r for r in recalls if not np.isnan(r)]
    
    mean_precision = np.mean(valid_precisions) if valid_precisions else 0.0
    mean_recall = np.mean(valid_recalls) if valid_recalls else 0.0
    
    return mean_precision, precisions, mean_recall, recalls


def calculate_fpr(pred: Union[torch.Tensor, np.ndarray],
                  target: Union[torch.Tensor, np.ndarray],
                  num_classes: int,
                  ignore_index: int = -1) -> Tuple[float, List[float]]:
    """
    Calculate False Positive Rate for each class
    
    Args:
        pred: Predicted labels
        target: Ground truth labels
        num_classes: Number of classes
        ignore_index: Label index to ignore
    
    Returns:
        Tuple of (mean_fpr, per_class_fpr)
    """
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Flatten arrays
    pred = pred.flatten()
    target = target.flatten()
    
    # Remove ignored indices
    if ignore_index >= 0:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]
    
    fprs = []
    
    for cls in range(num_classes):
        # False Positives and True Negatives
        fp = ((pred == cls) & (target != cls)).sum()
        tn = ((pred != cls) & (target != cls)).sum()
        
        # FPR: FP / (FP + TN)
        if fp + tn == 0:
            fprs.append(0.0)
        else:
            fprs.append(float(fp) / float(fp + tn))
    
    mean_fpr = np.mean(fprs)
    return mean_fpr, fprs


def calculate_f1_score(precision: Union[float, List[float]],
                       recall: Union[float, List[float]]) -> Union[float, List[float]]:
    """
    Calculate F1 score from precision and recall
    
    Args:
        precision: Precision value(s)
        recall: Recall value(s)
    
    Returns:
        F1 score(s)
    """
    if isinstance(precision, list) and isinstance(recall, list):
        f1_scores = []
        for p, r in zip(precision, recall):
            if np.isnan(p) or np.isnan(r) or (p + r) == 0:
                f1_scores.append(float('nan'))
            else:
                f1_scores.append(2 * p * r / (p + r))
        return f1_scores
    else:
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def calculate_dice_coefficient(pred: Union[torch.Tensor, np.ndarray],
                               target: Union[torch.Tensor, np.ndarray],
                               num_classes: int,
                               smooth: float = 1e-6) -> Tuple[float, List[float]]:
    """
    Calculate Dice coefficient (similar to F1 score for segmentation)
    
    Args:
        pred: Predicted labels
        target: Ground truth labels
        num_classes: Number of classes
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Tuple of (mean_dice, per_class_dice)
    """
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    pred = pred.flatten()
    target = target.flatten()
    
    dice_scores = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls).astype(float)
        target_cls = (target == cls).astype(float)
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)
    
    mean_dice = np.mean(dice_scores)
    return mean_dice, dice_scores


def calculate_confusion_matrix(pred: Union[torch.Tensor, np.ndarray],
                              target: Union[torch.Tensor, np.ndarray],
                              num_classes: int,
                              normalize: bool = False) -> np.ndarray:
    """
    Calculate confusion matrix
    
    Args:
        pred: Predicted labels
        target: Ground truth labels
        num_classes: Number of classes
        normalize: Whether to normalize the matrix
    
    Returns:
        Confusion matrix
    """
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    pred = pred.flatten()
    target = target.flatten()
    
    # Calculate confusion matrix
    cm = sklearn_confusion_matrix(target, pred, labels=list(range(num_classes)))
    
    if normalize:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
    
    return cm


def calculate_class_weights(labels: Union[torch.Tensor, np.ndarray],
                           num_classes: int,
                           method: str = 'inverse_frequency') -> np.ndarray:
    """
    Calculate class weights for handling class imbalance
    
    Args:
        labels: Ground truth labels from dataset
        num_classes: Number of classes
        method: Weighting method ('inverse_frequency', 'effective_number', 'balanced')
    
    Returns:
        Class weights array
    """
    # Convert to numpy if needed
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    labels = labels.flatten()
    
    # Count frequency of each class
    class_counts = np.bincount(labels, minlength=num_classes).astype(float)
    
    if method == 'inverse_frequency':
        # Inverse frequency weighting
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * num_classes
        
    elif method == 'effective_number':
        # Effective number of samples
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + 1e-6)
        weights = weights / weights.sum() * num_classes
        
    elif method == 'balanced':
        # Balanced weighting (sklearn style)
        total_samples = len(labels)
        weights = total_samples / (num_classes * class_counts + 1e-6)
        
    else:
        # No weighting
        weights = np.ones(num_classes)
    
    return weights


class MetricTracker:
    """
    Class to track and accumulate metrics during training/validation
    """
    
    def __init__(self, num_classes: int, ignore_index: int = -1):
        """
        Initialize metric tracker
        
        Args:
            num_classes: Number of classes
            ignore_index: Label index to ignore
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_correct = 0
        self.total_samples = 0
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.per_class_tp = np.zeros(self.num_classes)
        self.per_class_fp = np.zeros(self.num_classes)
        self.per_class_fn = np.zeros(self.num_classes)
        self.per_class_tn = np.zeros(self.num_classes)
    
    def update(self, pred: Union[torch.Tensor, np.ndarray],
               target: Union[torch.Tensor, np.ndarray]):
        """
        Update metrics with batch predictions
        
        Args:
            pred: Predicted labels
            target: Ground truth labels
        """
        # Convert to numpy if needed
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        pred = pred.flatten()
        target = target.flatten()
        
        # Remove ignored indices
        if self.ignore_index >= 0:
            mask = target != self.ignore_index
            pred = pred[mask]
            target = target[mask]
        
        # Update overall accuracy
        self.total_correct += (pred == target).sum()
        self.total_samples += len(target)
        
        # Update confusion matrix
        for t, p in zip(target, pred):
            if t < self.num_classes and p < self.num_classes:
                self.confusion_matrix[t, p] += 1
        
        # Update per-class statistics
        for cls in range(self.num_classes):
            self.per_class_tp[cls] += ((pred == cls) & (target == cls)).sum()
            self.per_class_fp[cls] += ((pred == cls) & (target != cls)).sum()
            self.per_class_fn[cls] += ((pred != cls) & (target == cls)).sum()
            self.per_class_tn[cls] += ((pred != cls) & (target != cls)).sum()
    
    def get_metrics(self) -> Dict:
        """
        Calculate all metrics from accumulated statistics
        
        Returns:
            Dictionary containing all metrics
        """
        # Overall accuracy
        accuracy = self.total_correct / (self.total_samples + 1e-6)
        
        # Per-class IoU
        ious = []
        for cls in range(self.num_classes):
            intersection = self.per_class_tp[cls]
            union = self.per_class_tp[cls] + self.per_class_fp[cls] + self.per_class_fn[cls]
            if union == 0:
                ious.append(float('nan'))
            else:
                ious.append(intersection / union)
        
        # Per-class precision
        precisions = []
        for cls in range(self.num_classes):
            tp_fp = self.per_class_tp[cls] + self.per_class_fp[cls]
            if tp_fp == 0:
                precisions.append(float('nan'))
            else:
                precisions.append(self.per_class_tp[cls] / tp_fp)
        
        # Per-class recall
        recalls = []
        for cls in range(self.num_classes):
            tp_fn = self.per_class_tp[cls] + self.per_class_fn[cls]
            if tp_fn == 0:
                recalls.append(float('nan'))
            else:
                recalls.append(self.per_class_tp[cls] / tp_fn)
        
        # Per-class FPR
        fprs = []
        for cls in range(self.num_classes):
            fp_tn = self.per_class_fp[cls] + self.per_class_tn[cls]
            if fp_tn == 0:
                fprs.append(0.0)
            else:
                fprs.append(self.per_class_fp[cls] / fp_tn)
        
        # Calculate means (ignoring nan)
        valid_ious = [x for x in ious if not np.isnan(x)]
        valid_precisions = [x for x in precisions if not np.isnan(x)]
        valid_recalls = [x for x in recalls if not np.isnan(x)]
        
        mean_iou = np.mean(valid_ious) if valid_ious else 0.0
        mean_precision = np.mean(valid_precisions) if valid_precisions else 0.0
        mean_recall = np.mean(valid_recalls) if valid_recalls else 0.0
        mean_fpr = np.mean(fprs)
        
        # F1 scores
        f1_scores = calculate_f1_score(precisions, recalls)
        valid_f1 = [x for x in f1_scores if not np.isnan(x)]
        mean_f1 = np.mean(valid_f1) if valid_f1 else 0.0
        
        return {
            'accuracy': accuracy,
            'mean_iou': mean_iou,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_fpr': mean_fpr,
            'mean_f1': mean_f1,
            'per_class_iou': ious,
            'per_class_precision': precisions,
            'per_class_recall': recalls,
            'per_class_fpr': fprs,
            'per_class_f1': f1_scores,
            'confusion_matrix': self.confusion_matrix
        }


def robustness_score(metrics: Dict, 
                     alpha: float = 0.5,
                     beta: float = 0.5) -> float:
    """
    Calculate a custom robustness score for SENTINEL system
    Combines mIoU and FPR reduction for adverse conditions
    
    Args:
        metrics: Dictionary containing metrics
        alpha: Weight for mIoU
        beta: Weight for FPR reduction
    
    Returns:
        Robustness score
    """
    miou = metrics.get('mean_iou', 0.0)
    fpr = metrics.get('mean_fpr', 1.0)
    
    # Normalize weights
    alpha = alpha / (alpha + beta)
    beta = beta / (alpha + beta)
    
    # Calculate score (higher is better)
    score = alpha * miou + beta * (1.0 - fpr)
    
    return score


def hallucination_rate(pred: Union[torch.Tensor, np.ndarray],
                       confidence: Optional[Union[torch.Tensor, np.ndarray]] = None,
                       threshold: float = 0.5) -> float:
    """
    Calculate hallucination rate (objects predicted with high confidence but incorrect)
    
    Args:
        pred: Predicted labels
        confidence: Confidence scores for predictions
        threshold: Confidence threshold for considering hallucination
    
    Returns:
        Hallucination rate
    """
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if confidence is not None and isinstance(confidence, torch.Tensor):
        confidence = confidence.cpu().numpy()
    
    if confidence is None:
        # Without confidence, can't calculate hallucination rate
        return 0.0
    
    # High confidence predictions
    high_conf_mask = confidence > threshold
    
    # Count high confidence predictions that are non-background
    high_conf_objects = (pred > 0) & high_conf_mask
    
    # This is a simplified metric - in practice, you'd need ground truth
    # to determine true hallucinations
    hallucination_ratio = high_conf_objects.sum() / (high_conf_mask.sum() + 1e-6)
    
    return hallucination_ratio


def print_metrics(metrics: Dict, 
                  class_names: Optional[List[str]] = None,
                  detailed: bool = True):
    """
    Pretty print metrics
    
    Args:
        metrics: Dictionary containing metrics
        class_names: Optional list of class names
        detailed: Whether to print per-class metrics
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(metrics.get('per_class_iou', [])))]
    
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    # Overall metrics
    print("\nOverall Metrics:")
    print(f"  Accuracy:        {metrics.get('accuracy', 0):.4f}")
    print(f"  Mean IoU:        {metrics.get('mean_iou', 0):.4f}")
    print(f"  Mean Precision:  {metrics.get('mean_precision', 0):.4f}")
    print(f"  Mean Recall:     {metrics.get('mean_recall', 0):.4f}")
    print(f"  Mean FPR:        {metrics.get('mean_fpr', 0):.4f}")
    print(f"  Mean F1:         {metrics.get('mean_f1', 0):.4f}")
    
    if 'robustness_score' in metrics:
        print(f"  Robustness:      {metrics['robustness_score']:.4f}")
    
    # Per-class metrics
    if detailed and 'per_class_iou' in metrics:
        print("\nPer-Class Metrics:")
        print("-"*60)
        print(f"{'Class':<20} {'IoU':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-"*60)
        
        for i, name in enumerate(class_names):
            iou = metrics['per_class_iou'][i]
            precision = metrics['per_class_precision'][i] if 'per_class_precision' in metrics else 0
            recall = metrics['per_class_recall'][i] if 'per_class_recall' in metrics else 0
            f1 = metrics['per_class_f1'][i] if 'per_class_f1' in metrics else 0
            
            if not np.isnan(iou):
                print(f"{name:<20} {iou:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
            else:
                print(f"{name:<20} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    print("="*60)


# Export main functions
__all__ = [
    'calculate_iou',
    'calculate_accuracy',
    'calculate_precision_recall',
    'calculate_fpr',
    'calculate_f1_score',
    'calculate_dice_coefficient',
    'calculate_confusion_matrix',
    'calculate_class_weights',
    'MetricTracker',
    'robustness_score',
    'hallucination_rate',
    'print_metrics'
]