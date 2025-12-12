# SENTINEL: Semantic Enhancement Through Intelligent Noise Elimination and Labeling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/cuda-enabled-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SENTINEL (Semantic Enhancement Through Intelligent Noise Elimination and Labeling) is a LiDAR-only semantic segmentation system for autonomous driving. It combines a PointNet++ backbone with a post-processing geometric refinement module to reduce “ghost” detections and expose how performance degrades with distance.

The code and experiments here correspond to the distance-aware point cloud segmentation.

---

## 1. Core Ideas

- **Distance-stratified evaluation**  
  Instead of a single scene-wide mIoU, SENTINEL evaluates performance in radial bins:
  `0–10 m, 10–20 m, 20–30 m, 30–40 m, 40–50 m, 50–70 m, 70–100 m`.  
  This reveals a clear “knee” in performance around 35–40 m that is invisible in aggregate metrics.

- **Hybrid architecture**  
  Stage 1: PointNet++ semantic segmentation on raw point clouds.  
  Stage 2: RANSAC-based geometric validation that enforces simple physical constraints
  (planarity, bounding-box size, aspect ratio, ground contact).

- **Cheap geometric “gatekeeper”**  
  In the C++ deployment, the geometric refinement adds only ~12–13 ms per frame on an NVIDIA T4,
  while the whole pipeline is ~1.05 s per frame. The bottleneck is the backbone and data marshalling,
  not the geometric checks.

- **Realistic, slightly painful honesty**  
  The system is not real-time yet (~1 Hz), but the method shows that geometry can significantly
  reshape distance-performance behavior at negligible extra cost.

---

## 2. Key Results (SemanticKITTI, Sequence 08 Validation)

All numbers below are from the final thesis version.

### 2.1 Global metrics

- **Baseline (PointNet++) mIoU**: 0.482  
- **SENTINEL (PointNet++ + geometry) mIoU**: 0.547  
- **Absolute mIoU gain**: +0.065  
- **False Positive Rate reduction**: ~6–7 percentage points across distance bands  
  (roughly ~40% relative reduction in “ghost” detections)

### 2.2 Distance-stratified performance (illustrative)

Using three merged bands for readability:

| Distance Range (m) | Baseline mIoU | SENTINEL mIoU | Δ mIoU | Baseline FPR | SENTINEL FPR | Δ FPR  |
|--------------------|--------------:|--------------:|:------:|-------------:|-------------:|:------:|
| 0–20               | 0.651         | 0.731         | +0.080 | 0.145        | 0.082        | −0.063 |
| 20–50              | 0.286         | 0.413         | +0.127 | 0.162        | 0.093        | −0.069 |
| 50–100*            | 0.089         | 0.214         | +0.125 | 0.181        | 0.107        | −0.074 |

\*Far-field bins (50–70 m, 70–100 m) are merged due to very few labeled objects at those ranges.

### 2.3 Latency (C++ / LibTorch deployment, NVIDIA T4)

| Configuration          | Mean Latency (ms) | 99th Percentile (ms) | FPS  |
|------------------------|------------------:|----------------------:|:----:|
| Python, CPU            | 7904.7           | 10374.9              | 0.1  |
| PyTorch, GPU           | 2634.9           | 3458.3               | 0.4  |
| TorchScript, GPU       | 2627.3           | 3436.3               | 0.4  |
| C++ / LibTorch, GPU    | 1054.0           | 1383.3               | 0.9  |

- Geometric refinement itself: **~12–13 ms** per frame (≈1% of total runtime).  
- Main bottlenecks: PointNet++ forward pass + CPU↔GPU marshalling.

---

## 3. Repository Layout

project-sentinel/
├── setup/                      # Environment / config helpers
│   ├── 00_setup_environment.sh
│   ├── 01_config.yaml
│   └── 02_requirements.txt
│
├── data_preparation/           # Dataset download and preprocessing
│   ├── 03_download_from_gcs.py
│   ├── 04_preprocess_data.py
│   ├── 05_distance_analysis.py
│   └── 06_data_augmentation.py
│
├── src/
│   ├── python/                 # Training / evaluation code (PointNet++)
│   │   ├── models/             # PointNet++ segmentation, loss functions
│   │   ├── datasets/           # SemanticKITTI dataset loaders
│   │   ├── utils/              # Metrics, logging, visualization
│   │   └── config/             # Training configs
│   │
│   └── cpp/                    # C++ deployment and geometric refinement
│       ├── include/            # C++ headers
│       ├── src/                # C++ implementations (LibTorch + PCL)
│       └── CMakeLists.txt
│
├── notebooks/                  # Colab / Jupyter experiments (optional)
│   ├── 01_data_overview.ipynb
│   ├── 02_density_statistics.ipynb
│   ├── 03_pointnet_training.ipynb
│   ├── 04_distance_stratified_eval.ipynb
│   └── 05_visualization.ipynb
│
├── scripts/                    # Convenience scripts
│   ├── run_training.sh
│   ├── run_evaluation.sh
│   ├── build_cpp.sh
│   └── download_semantickitti.sh
│
├── models/                     # Saved checkpoints (ignored in git)
├── docs/                       #  PDF, chapter drafts, figures
└── README.md

4. Getting Started
4.1 Requirements

Python 3.8+

PyTorch 2.x with CUDA support

CMake and a C++17 compiler (for deployment)

PCL (Point Cloud Library) and Eigen (for geometric refinement)

Access to the SemanticKITTI dataset
4.2 Setup


Create and activate a virtual environment (strongly recommended):

python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate


Install Python dependencies:

pip install -r setup/02_requirements.txt


(Optional) Configure environment variables and paths in setup/01_config.yaml
for dataset location, logging directories, and GPU selection.



5. Data Preparation (SemanticKITTI)

Download SemanticKITTI according to the official instructions and place the raw .bin
and label files under a directory, for example:

/data/semantickitti/
    ├── dataset/
    │   ├── sequences/00/velodyne/*.bin
    │   ├── sequences/00/labels/*.label
    │   └── ...


Run the preprocessing pipeline to generate training metadata, block partitions,
and distance statistics:

python data_preparation/04_preprocess_data.py \
    --dataset_root /data/semantickitti/dataset \
    --output_root  /data/semantickitti/processed


Optionally compute the distance-binned statistics and tables used in the thesis:

python data_preparation/05_distance_analysis.py \
    --processed_root /data/semantickitti/processed


6. Training the Baseline (PointNet++)

A typical training command (adapt to your actual script and config names):

python -m src.python.train_pointnet2 \
    --config src/python/config/pointnet2_semantickitti.yaml \
    --data_root /data/semantickitti/processed \
    --log_dir runs/pointnet2_baseline


The baseline is trained on SemanticKITTI sequences 00–07, 09–10, with sequence 08 used
exclusively for validation, matching the experimental setup in the thesis.



7. Running SENTINEL (Hybrid Model)
7.1 Python evaluation (research mode)

After training the backbone, you can evaluate the hybrid system using the distance-stratified
pipeline:

python -m src.python.evaluate_sentinel \
    --config src/python/config/pointnet2_semantickitti.yaml \
    --checkpoint models/pointnet2_final.pth \
    --data_root /data/semantickitti/processed \
    --output_dir results/sentinel_eval


This will:

run the PointNet++ backbone on Sequence 08,

apply the geometric refinement module,

accumulate per-bin confusion matrices,

output JSON/CSV summaries and optional plots.


7.2 C++ deployment (LibTorch + PCL)

Export the trained model to TorchScript:

python -m src.python.export_to_torchscript \
    --checkpoint models/pointnet2_final.pth \
    --output models/pointnet2_ts.pt


Build the C++ project:

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)


Run the deployed pipeline on Sequence 08:

./sentinel_deploy \
    --model ../models/pointnet2_ts.pt \
    --sequence_root /data/semantickitti/dataset/sequences/08 \
    --output_dir ../results/deploy_seq08


This reproduces the end-to-end latency numbers reported in the thesis (≈1.05 s per frame).



8. Reproducing Thesis Figures and Tables

The following scripts/notebooks are intended to regenerate the main analysis artifacts:

Distance-stratified mIoU and FPR curves (Chapter 4.1)
notebooks/04_distance_stratified_eval.ipynb or a dedicated evaluate_distance_bins.py.

Car statistics by range (Table 3.1)
data_preparation/05_distance_analysis.py.

Latency comparison (Table 3.3)
C++ deployment benchmark built into sentinel_deploy (e.g., --profile flag).

Qualitative examples of geometric refinement (failure modes)
Visualization helpers in src/python/utils/vis.py or notebooks under notebooks/.



9. Limitations and Caveats

A README is not a marketing brochure, so here are the important warts up front:

The system is not real-time in its current form (~1 Hz on a T4).

Distance-stratified far-field statistics are limited by very few labeled objects beyond ~70 m.

Geometric constraints are hand-crafted and tuned for typical passenger cars; they can
over-reject atypical vehicles and very sparse pedestrians.

Adverse weather robustness is evaluated with simple synthetic perturbations, not real
multi-weather LiDAR data.

