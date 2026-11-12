# Part 1: Training, Validation, and Benchmark Suite

## Overview
This directory contains the complete end-to-end training, validation, and benchmarking pipeline for botanical pathological classification on the PlantVillage dataset across 15 Solanaceous and Bell Pepper classes. The pipeline encompasses rigorous hash-based perceptual deduplication (zero-leakage splits), photometric exploratory data analysis, three architectural variants trained under identical regularization protocols, and quantitative evaluation across accuracy, per-class macro F1, parameter efficiency, and runtime inference latency.

---

## Directory Contents

| Component | Path | Description |
| :--- | :--- | :--- |
| **Data Preprocessing** | `DataPreproccessing.ipynb` | Raw dataset ingestion, label normalization, metadata indexing, and initial train/val/test partitioning. |
| **Deduplication Engine** | `CleanUp-TestSet.ipynb` | Perceptual difference hashing (`dHash`, Hamming threshold <= 6) to eliminate near-duplicate specimens between train and test splits, guaranteeing strictly zero data leakage. |
| **Exploratory Data Analysis** | `EDA on PlantVillage.ipynb` | Taxonomic class distribution, Pareto analysis, photometric channel distributions (RGB, HSV, Excess Green Index ExG), and statistical extreme detection. |
| **EDA Visualizations** | `EDA Graphs/` | Publication-ready visual artifacts, box plots, class feature heatmaps, and outlier summary tables. |
| **Model Training & Benchmarking** | `Models.ipynb` | PyTorch training harnesses for the three comparative architectural variants, FP16 quantization, and evaluation pipelines. |
| **Dataset Utilities** | `dataset_utils.py` | PyTorch `Dataset` loaders, albumentation/torchvision augmentation transforms, and deterministic sample indexing. |
| **Result Inspector** | `view_results.py` | Automated CLI report generator that aggregates JSON training logs into tabular LaTeX/Markdown formats. |
| **Comparison Dashboard** | `model_comparison_dashboard.png` | Multi-panel visual comparison showing loss convergence, validation accuracy curves, per-class F1 distribution, and latency vs. accuracy Pareto frontier. |
| **Evaluation Metrics** | `final_model_comparison_metrics.csv` | Macro metrics: Accuracy, Precision, Recall, Macro F1, Loss, and Inference Time. |
| **Stability & Complexity** | `final_model_stability_and_complexity.csv` | Parameter counts, FLOPs, model footprint, and epoch-to-epoch variance. |
| **Per-Class F1 Matrix** | `final_per_class_f1_comparison.csv` | Granular per-pathology F1 scores across all 15 diagnostic classes. |
| **Experiment Logs** | `logs/` | Raw JSON epoch telemetry (`results_se_block.json`, `results_conv_bn_relu.json`, `results_cbam.json`) and raw execution log (`training_and_benchmark.log`). |

---

## Architectural Comparison Matrix

Three deep convolutional architectures with transfer learning backbones were systematically trained and evaluated under strictly controlled hyperparameter configurations (Cross-Entropy loss, AdamW optimizer, cosine annealing learning rate scheduler, batch size 64, image resolution 224x224):

| Metric | Baseline Conv-BN-ReLU | Squeeze-and-Excitation (SE-Block) | CBAM (Dual Channel + Spatial) |
| :--- | :---: | :---: | :---: |
| **Backbone** | ResNet-50 | ResNet-50 | ResNet-50 |
| **Attention Mechanism** | None (Dense Head) | Channel Attention (SE) | Channel + Spatial Attention |
| **Test Accuracy** | 99.11% | **99.39%** | 99.28% |
| **Macro F1 Score** | 0.9887 | **0.9924** | 0.9909 |
| **Parameter Count** | 25.56 M | 25.58 M | 25.60 M |
| **FP16 Checkpoint** | 63.27 MB | 63.41 MB | 63.41 MB |
| **CPU Latency (per batch)** | 14.8 ms | 15.2 ms | 17.6 ms |
| **Convergence Stability** | Baseline | **Superior (Lowest Variance)** | High |

**Champion Architecture Selection**: The Squeeze-and-Excitation (SE-Block) variant achieved the highest test accuracy (99.39%) and Macro F1 (0.9924) with near-zero latency overhead (+0.4 ms) relative to the baseline, outperforming the computationally heavier CBAM head.

---

## Experimental Protocol & Data Hygiene

### 1. Hash-Based Deduplication (Zero Data Leakage)
In agricultural datasets such as PlantVillage, camera burst shots often result in duplicate frames appearing across both training and test partitions. To prevent synthetic metric inflation:
- Perceptual difference hashing (`dHash`) computed a 64-bit fingerprint for every sample.
- Pairs exhibiting Hamming distance <= 6 were flagged as near-duplicates.
- Duplicate clusters were isolated to preserve test split integrity, ensuring all reported metrics reflect true generalized out-of-distribution performance.

### 2. Training Regularization
- **Photometric Augmentations**: Random horizontal/vertical flips, affine rotations (-15 deg to +15 deg), color jitter (brightness 0.2, contrast 0.2, saturation 0.2), and cutout.
- **Normalization**: Standard ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- **FP16 Half-Precision Optimization**: Model checkpoints were quantized to half-precision float16 (`.pth` weights under 65 MB) for cross-platform local CPU and edge deployment without requiring GPU hardware.

---

## How to Run & Reproduce

### 1. Environment Setup
Activate your Python environment (Python 3.9 - 3.11 recommended):
```bash
pip install -r requirements.txt
```

### 2. Inspecting Existing Benchmark Telemetry
To inspect the benchmark metrics directly from terminal:
```bash
python3 view_results.py
```

### 3. Executing Notebooks
Launch Jupyter Lab or Notebook:
```bash
jupyter lab
```
Run the notebooks in sequential order:
1. `DataPreproccessing.ipynb` - Dataset structuring and verification.
2. `CleanUp-TestSet.ipynb` - Perceptual hash deduplication audit.
3. `EDA on PlantVillage.ipynb` - Exploratory analysis and photometric distribution profiling.
4. `Models.ipynb` - Model training, loss curve tracking, and validation evaluation.
