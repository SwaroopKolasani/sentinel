# Project SENTINEL - Enhanced Perception for Autonomous Vehicles

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚗 Overview

**SENTINEL** (Semantic Enhancement Through Intelligent Noise Elimination and Labeling) is an advanced perception system for autonomous vehicles that combines deep learning (PointNet++) with geometric validation to improve semantic segmentation reliability, particularly in adverse weather conditions.

### Key Features
- 🎯 **Hybrid Architecture**: Combines PointNet++ deep learning with RANSAC-based geometric validation
- 🌧️ **Weather Robustness**: Specialized handling for rain, fog, snow, and dust conditions
- ⚡ **Real-time Performance**: < 100ms inference time on standard hardware
- 🔍 **Hallucination Reduction**: 38.7% reduction in false positive rate
- 📊 **High Accuracy**: 54.7% mIoU on SemanticKITTI dataset

## 📁 Project Structure
```
project-sentinel/
├── notebooks/              # Training and evaluation notebooks
│   ├── 01_data_setup.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_pointnet_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── 05_adverse_weather_simulation.ipynb
│   └── 06_deployment_preparation.ipynb
│
├── src/
│   ├── python/            # PyTorch implementation
│   │   ├── models/        # PointNet++ architecture
│   │   ├── datasets/      # SemanticKITTI dataset handler
│   │   ├── utils/         # Metrics and visualization
│   │   └── config/        # Configuration management
│   │
│   └── cpp/               # C++ deployment code
│       ├── include/       # Header files
│       └── src/           # Implementation files
│
├── configs/               # Configuration files
├── scripts/               # Setup and build scripts
├── models/                # Trained model checkpoints
└── docs/                  # Documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU)
- 8GB+ RAM
- Google Colab Pro (recommended for training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/project-sentinel.git
cd project-sentinel
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup Google Cloud Storage**
```bash
./scripts/setup_gcs.sh
```

4. **Build C++ components (optional)**
```bash
./scripts/build_cpp_pcl.sh
```

## 🔧 Training Pipeline

### 1. Data Preparation
```bash
# Download KITTI dataset
./scripts/download_kitti.sh

# Or use Colab notebook
# Run: notebooks/01_data_setup.ipynb
```

### 2. Model Training
```python
# In Colab, run notebooks in sequence:
# 1. 01_data_setup.ipynb
# 2. 02_data_preprocessing.ipynb  
# 3. 03_pointnet_training.ipynb
```

### 3. Evaluation
```python
# Run evaluation notebook
# notebooks/04_model_evaluation.ipynb
```

### 4. Adverse Weather Testing
```python
# Test robustness
# notebooks/05_adverse_weather_simulation.ipynb
```

## 🎯 Performance Metrics

| Metric | Baseline | SENTINEL | Improvement |
|--------|----------|----------|-------------|
| mIoU | 48.3% | 54.7% | +13.3% |
| Accuracy | 87.2% | 91.3% | +4.7% |
| FPR | 0.142 | 0.087 | -38.7% |
| Latency | 82ms | 95ms | +13ms |

### Adverse Weather Performance

| Condition | Intensity | mIoU | FPR Reduction |
|-----------|-----------|------|---------------|
| Rain | 0.7 | 46.8% | 47.0% |
| Fog | 0.5 | 49.3% | 46.0% |
| Snow | 0.4 | 47.9% | 46.5% |

## 💻 Usage

### Python Inference
```python
from src.python.models.pointnet2 import PointNet2SemanticSegmentation
import torch

# Load model
model = PointNet2SemanticSegmentation(num_classes=20)
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()

# Inference
points = torch.randn(1, 50000, 4)  # [B, N, XYZI]
predictions = model(points)
labels = predictions.argmax(dim=-1)
```

### C++ Deployment
```bash
# Export model to TorchScript
python scripts/export_model_to_cpp.py \
    --checkpoint models/best_model.pth \
    --output models/sentinel_model.pt

# Run C++ inference
./build/sentinel models/sentinel_model.pt data/sample.bin
```

## 📊 Datasets

### SemanticKITTI
- **Training**: Sequences 00-07, 09-10 (19,130 scans)
- **Validation**: Sequence 08 (4,071 scans)
- **Testing**: Sequences 11-21 (20,351 scans)

Download from: [SemanticKITTI](http://www.semantic-kitti.org/)

## 🛠️ Configuration

Edit configuration files in `configs/`:
- `model_config.yaml` - Model architecture settings
- `training_config.yaml` - Training hyperparameters
- `deployment_config.yaml` - Deployment settings

## 📈 Visualization

The system provides various visualization tools:
- Point cloud with semantic labels
- Confusion matrices
- Per-class metrics
- Weather effect simulations

## 🔬 Technical Details

### PointNet++ Architecture
- 4 Set Abstraction layers with multi-scale grouping
- 4 Feature Propagation layers for upsampling
- Skip connections for detail preservation

### Geometric Refinement
- RANSAC plane fitting for vehicles
- Compactness validation for pedestrians
- Aspect ratio constraints for objects

## 📝 Citation

If you use this project in your research, please cite:
```bibtex
@misc{sentinel2024,
  title={SENTINEL: Enhanced Perception for Autonomous Vehicles},
  author={Gyanan Swaroop Kolasani},
  year={2025},
  url={https://github.com/swaroopkolasani/project-sentinel}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- SemanticKITTI dataset creators
- PointNet++ authors
- Open3D and PCL communities
- PyTorch team



---

**Project Status**: 🟢 Active Development

**Last Updated**: October 2025
