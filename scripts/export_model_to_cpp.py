#!/usr/bin/env python3
"""
Export trained PyTorch model to TorchScript for C++ deployment
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.python.models.pointnet2 import PointNet2SemanticSegmentation


def export_model(checkpoint_path, output_path, batch_size=1, num_points=50000):
    """
    Export PyTorch model to TorchScript format
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pth)
        output_path: Path to save TorchScript model (.pt)
        batch_size: Batch size for tracing
        num_points: Number of points for tracing
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Load model
    model = PointNet2SemanticSegmentation(num_classes=20)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create example input for tracing
    example_input = torch.randn(batch_size, num_points, 4)  # XYZI
    
    print("Tracing model...")
    # Use tracing to convert to TorchScript
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)
    
    # Optimize for inference
    print("Optimizing for inference...")
    traced_model = torch.jit.optimize_for_inference(traced_model)
    
    # Save model
    print(f"Saving TorchScript model to: {output_path}")
    traced_model.save(output_path)
    
    # Verify the saved model
    print("Verifying saved model...")
    loaded_model = torch.jit.load(output_path)
    loaded_model.eval()
    
    # Test inference
    with torch.no_grad():
        test_output = loaded_model(example_input)
    
    print(f"Model output shape: {test_output.shape}")
    print(f"Expected shape: [{batch_size}, {num_points}, 20]")
    
    # Get file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model size: {file_size:.2f} MB")
    
    print("Export successful!")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export SENTINEL model to TorchScript')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to PyTorch checkpoint (.pth)')
    parser.add_argument('--output', type=str, default='models/sentinel_model.pt',
                       help='Output path for TorchScript model')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for tracing')
    parser.add_argument('--num-points', type=int, default=50000,
                       help='Number of points for tracing')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export model
    export_model(
        args.checkpoint,
        args.output,
        args.batch_size,
        args.num_points
    )


if __name__ == '__main__':
    main()