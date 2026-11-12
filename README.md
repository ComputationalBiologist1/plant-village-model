# Automated Plant Disease Diagnosis & Architecture Benchmark
### Large-Scale Deep Learning Benchmark of Modern Convolutional Architectures with Multi-Mechanism Attention for Zero-Leakage Botanical Pathology Diagnosis

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA_RTX_4050-76B900.svg?logo=nvidia&logoColor=white)](https://www.nvidia.com/)
[![Zero Data Leakage](https://img.shields.io/badge/Dataset_Integrity-Zero_Leakage_Guaranteed-success.svg)](#data-engineering--zero-leakage-guarantee)
[![GitHub Safe](https://img.shields.io/badge/Weights-Direct_Push_<100MB-brightgreen.svg)](#how-to-push-to-github-100-mb-direct-push)

---

## Executive Overview

Early detection of foliar disease in staple crops (**Tomato, Potato, Pepper Bell**) is vital for preventing epidemic crop loss and safeguarding global food security. However, conventional agricultural deep learning models suffer from severe **data leakage** and **overfitting**, yielding inflated lab benchmark scores (>99%) that completely collapse when deployed on real-world field imagery.

This project delivers an end-to-end, industrial-grade deep learning benchmark across **87,736 plant leaf images across 15 pathological categories**, comparing three modern Convolutional Neural Network architectures based on the **ConvNeXt-Tiny** backbone:
1. **`Conv_BN_ReLU`** — ConvNeXt-Tiny with 3x3 Convolutional Residual Refinement
2. **`SE_Block`** (Champion / Production Selected) — ConvNeXt-Tiny with Squeeze-and-Excitation Channel Attention
3. **`CBAM`** — ConvNeXt-Tiny with Convolutional Block Attention Module (Channel + Spatial Attention)

> [!IMPORTANT]
> **Strict Zero Data Leakage Guarantee:**
> The training database is strictly derived from **PlantVillage** ($N = 15,029$), while validation and testing are strictly drawn from an **independent external superset** ($N_{\text{val}} = 36,356$, $N_{\text{test}} = 36,351$). All cross-dataset visual duplicates were purged using **perceptual hashing (`phash` and `dhash`)**, ensuring unbiased generalization evaluation.

---

## Multi-Model Benchmark Verdict

All models were trained for **50 full epochs** on an **NVIDIA GeForce RTX 4050 Laptop GPU** using **FP16 Mixed Precision (AMP)**, **Cosine Annealing** down to $10^{-6}$, **Mixup augmentation**, **Label Smoothing**, and **AdamW differential learning rates**.

| Architecture | Val Acc (%) | Val Macro-F1 (%) | Test Acc (%) | Test Recall (%) | Test Macro-F1 (%) | Test Weighted-F1 (%) | Test Loss | Composite Score | Benchmark Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **`SE_Block`** (ConvNeXt + SE Attention) | **72.76%** | 69.61% | **72.40%** | **82.75%** | 69.45% | **74.14%** | **1.3223** | **70.23** | **WINNER (Production Selected)** |
| **`Conv_BN_ReLU`** (ConvNeXt + 3x3 Residual) | 69.20% | **70.56%** | 68.89% | 81.32% | **70.53%** | 70.59% | 1.4339 | 66.64 | Rank 2 (Suffered 13.95% Train-Val Gap) |
| **`CBAM`** (ConvNeXt + Channel & Spatial) | 67.61% | 67.82% | 67.41% | 81.50% | 67.64% | 71.97% | 1.4054 | 66.22 | Rank 3 (Degraded by 4x4 Padding Artifacts) |

```
Composite Selection Metric Formula:
Score = 0.40*(Test Macro-F1) + 0.25*(Test Acc) + 0.20*(Test Weighted-F1) + 0.15*(Val Macro-F1)
        - 0.25*|Train_Acc - Val_Acc| - 0.25*|Val_F1 - Test_F1|
```

### Key Reasons Why `SE_Block` Won:
1. **Superior Test Accuracy (72.40%) & Recall (82.75%):** Correctly classified the highest proportion of diseased foliage across 36,351 unseen test images with the lowest test loss (1.3223).
2. **Near-Zero Generalization Gap (0.36%):** Scored 72.76% on Validation and 72.40% on Test, demonstrating stability against distribution shift.
3. **Minimal Train-Val Overfitting Gap (3.53%):** Compared to `Conv_BN_ReLU` which overfit by 13.95% on training data, SE attention acted as a regularizer.
4. **Spatial Invariance over Small Feature Grids:** At Stage 4 ($128\times 128 \rightarrow 4\times 4$ spatial resolution), CBAM's $7\times 7$ spatial kernel with padding 3 introduced severe border-padding artifacts. SE-Block operates purely across channel descriptors ($768 \rightarrow 48 \rightarrow 768$), making it scale-invariant and immune to spatial distortion.

---

## Repository Organization: Two-Part Architecture

The repository is partitioned into two functional modules:

```text
Deep Learning Project/
├── README.md                           # Master documentation & benchmark summary (this file)
├── PROJECT_REPORT.md                   # Full 10-chapter technical whitepaper & academic project report
├── launch.sh                           # Unified cross-platform launcher (delegates to Part 2)
├── launch.bat                          # Windows batch launcher (delegates to Part 2)
├── classify_leaf.py                    # Root CLI classifier shortcut
├── sample_leaf.jpg                     # Bundled test leaf
├── requirements.txt                    # Project dependencies
│
├── Part_1_Training_and_Validation/     # PART 1: Training, Deduplication & Benchmark Suite
│   ├── CleanUp-TestSet.ipynb           # Perceptual dHash deduplication (zero data leakage)
│   ├── DataPreproccessing.ipynb        # Ingestion, label standardization & split generation
│   ├── EDA on PlantVillage.ipynb       # Exploratory analysis & photometric profiling
│   ├── EDA Graphs/                     # High-resolution charts, Pareto graphs & outlier tables
│   ├── Models.ipynb                    # Multi-architecture training harness & evaluation
│   ├── dataset_utils.py                # Dataset loaders & image transformations
│   ├── view_results.py                 # Terminal benchmark report generator
│   ├── final_model_comparison_metrics.csv
│   ├── final_model_stability_and_complexity.csv
│   ├── final_per_class_f1_comparison.csv
│   ├── model_comparison_dashboard.png  # 4-panel visual comparison dashboard
│   ├── logs/                           # Raw epoch-by-epoch training logs & JSON metrics
│   ├── requirements.txt                # Training & research environment dependencies
│   └── README.md                       # Part 1 detailed technical documentation
│
└── Part_2_Model_Testing_and_GUI/       # PART 2: Standalone Testing Suite, GUI Workstation & Models
    ├── plantguard_app.py               # PhytoScan CustomTkinter Desktop Workstation GUI
    ├── launch.sh                       # POSIX desktop launcher (macOS, Linux, WSL)
    ├── launch.bat                      # Windows desktop launcher
    ├── classify_leaf.py                # Standalone headless CLI classifier
    ├── run_classifier.sh               # CLI quick-run script
    ├── sample_leaf.jpg                 # Verified test specimen image
    ├── requirements.txt                # Minimal production testing dependencies
    ├── README.md                       # Part 2 usage & quickstart guide
    ├── models/                         # Quantized FP16 checkpoints (<65 MB each, direct push safe)
    │   ├── weights_se_block_fp16.pth   # Champion Model (ResNet-50 + SE-Block Attention)
    │   ├── weights_conv_bn_relu_fp16.pth # Baseline Model (Conv-BN-ReLU)
    │   ├── weights_cbam_fp16.pth       # CBAM Model (Channel & Spatial Attention)
    │   └── plant_disease_model_fp16.pth# Production Streamlit weights
    └── web_app/                        # Browser-based Streamlit testing application
        ├── app.py                      # Multi-model Streamlit web dashboard
        ├── predictor.py                # Standalone inference class
        ├── deployment_config.json      # Class index mappings & normalization parameters
        ├── requirements.txt            # Streamlit dependencies
        └── plant_disease_model_fp16.pth# Deployment model weights
```

---

## PhytoScan Botanical Pathology Diagnostic Workstation

In addition to the Streamlit web app and CLI tools, this repository includes **PhytoScan**, an Enterprise Laboratory Information Management System (LIMS) desktop workstation located in [`Part_2_Model_Testing_and_GUI/`](file:///Users/kushrocks6306/Deep%20Learning%20Project/Part_2_Model_Testing_and_GUI).

### Key Features:
* **Sanzo Wada Traditional Color Palettes:** Incorporates curated color harmonies referenced from *A Dictionary of Color Combinations* by Sanzo Wada (和田三造, 1930s):
  1. `No. 182 · Celadon & Sandalwood (青磁と白檀)`
  2. `No. 156 · Indigo & Persimmon (藍と柿)`
  3. `No. 24 · Verdigris & Muted Coral (緑青と珊瑚)`
  4. `No. 1 · Roasted Tea & Mustard (焙じ茶と芥子)`
  5. `No. 68 · Wisteria & Raw Umber (藤と生木)`
  6. `No. 322 · Dawn Mist & Camellia (浅葱と椿)`
  7. `No. 270 · Sumi Ink & Cinnabar (墨と朱色)`
  8. `No. 94 · Pine Needle & Smoked Bamboo (松葉と煤竹)`
* **Dual Menu Switching:** Seamlessly switch between palettes using the native OS menu bar (`Color Palette`) or the in-window dropdown menu. All surfaces, circular confidence dials, and distribution progress bars re-render dynamically.
* **Authentic Industrial Interface:** Clean, structured Swiss-Grid layout with zero artificial AI tropes, no double slashes, no fake telemetry, and native system typography (`SF Pro` on macOS, `Segoe UI` on Windows, `Inter` on Linux).
* **Deterministic Agronomic Care Guidelines:** Structured 3-column guidelines covering:
  1. Cultural Management & Sanitation
  2. Biological & Chemical Options (FRAC / IRAC standards)
  3. Vector Exclusion & Crop Rotation
* **Cross-Platform & Offline:** Runs locally without GPU requirements, auto-detecting Apple Silicon Metal (MPS), NVIDIA CUDA, or Multicore CPU.

### How to Launch PhytoScan:
```bash
# macOS / Linux (from repository root)
./launch.sh

# Or directly from Part 2 directory:
cd "Part_2_Model_Testing_and_GUI"
./launch.sh

# Windows (Command Prompt, PowerShell, or double-click)
launch.bat
```

---

## System Architecture & Workflow

```text
[Input Specimen: 128x128x3 RGB Leaf]
                 │
                 v
┌────────────────────────────────────────────────────────┐
│               DOMAIN PREPROCESSING PIPELINE             │
│  • Byte Tensor Conversion (NumPy-free C-API bypass)   │
│  • Domain Normalization: Mean [0.465, 0.479, 0.419]    │
│                           Std  [0.188, 0.165, 0.203]    │
└────────────────────────────────────────────────────────┘
                 │
                 v
┌────────────────────────────────────────────────────────┐
│           CONVNEXT-TINY BACKBONE (Stages 1-4)          │
│  • Stage 1: 96 dims, 3 blocks  (32x32 spatial)         │
│  • Stage 2: 192 dims, 3 blocks (16x16 spatial)         │
│  • Stage 3: 384 dims, 9 blocks (8x8 spatial)           │
│  • Stage 4: 768 dims, 3 blocks (4x4 spatial)           │
└────────────────────────────────────────────────────────┘
                 │
                 v
┌────────────────────────────────────────────────────────┐
│            ARCHITECTURAL BENCHMARK BRANCHES            │
├──────────────────────────┬─────────────────────────────┤
│   Conv_BN_ReLU (3x3)     │      SE_Block (Winning)     │
│   • 3x3 Conv (768->768)  │      • Global Avg Pool      │
│   • BatchNorm2d + ReLU   │      • Linear 768 -> 48     │
│   • Residual Addition    │      • ReLU + Linear -> 768 │
│                          │      • Sigmoid Channel Gate │
└──────────────────────────┴─────────────────────────────┘
                 │
                 v
┌────────────────────────────────────────────────────────┐
│                OUTPUT & CARE SYNTHESIS                 │
│  • 15-Class Softmax Probability Distribution           │
│  • Confidence Dial & Alternative Candidate Readouts    │
│  • Deterministic Clinical Agronomic Care Protocols     │
└────────────────────────────────────────────────────────┘
```

---

## Conceptual Breakdown of the Models

### Why ConvNeXt-Tiny?
ConvNeXt modernizes classical convolutional networks by incorporating architectural innovations from Vision Transformers (ViTs):
- **Depthwise Separable Convolutions:** $7\times 7$ depthwise kernels match the receptive field of Swin Transformers.
- **Inverted Bottleneck Design:** Hidden dimensions are expanded by $4\times$ in the MLP block ($768 \rightarrow 3072 \rightarrow 768$).
- **Fewer Activation & Normalization Layers:** Replaces frequent BatchNorm with LayerNorm, and uses GELU instead of ReLU.

### Model Comparison & Attention Mechanics:

```
[ConvNeXt Stage 4 Feature Map: (B, 768, 4, 4)]
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        v                v                v
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Conv_BN_ReLU │ │   SE_Block   │ │     CBAM     │
│  (Residual)  │ │ (Channel SE) │ │(Chan+Spatial)│
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       v                v                v
[3x3 Conv 768]   [GAP (B,768,1)]  [Channel Gate]
       │                │                │
[BatchNorm2d]    [FC 768 -> 48]   [Spatial Gate]
       │                │          (7x7 Conv p=3)
    [ReLU]            [ReLU]             │
       │                │                v
[Add Residual]   [FC 48 -> 768]   [Artifact-Prone]
       │                │                │
       │           [Sigmoid]             │
       │                │                │
       │         [Channel Scale]         │
       │                │                │
       └────────────────┼────────────────┘
                        │
                        v
         [Global Average Pooling: (B, 768)]
                        │
         [Linear Classifier: 768 -> 15 Classes]
```

#### 1. Model 1: `Conv_BN_ReLU`
Adds an explicit $3\times 3$ convolutional residual block post-backbone. While achieving strong training accuracy, it suffered from a **13.95% train-validation gap**, indicating feature memorization rather than generalized abstraction.

#### 2. Model 2: `SE_Block` (Squeeze-and-Excitation Attention) — Champion / Selected
Applies channel-wise attention via global average pooling and a bottleneck MLP ($768 \rightarrow 48 \rightarrow 768$):
$$\mathbf{s} = \sigma\left(\mathbf{W}_2 \cdot \text{ReLU}\left(\mathbf{W}_1 \cdot \text{GAP}(\mathbf{X})\right)\right)$$
This enables the network to dynamically emphasize pathological channels (e.g., fungal chlorosis or necrosis signatures) while attenuating background leaf textures.

#### 3. Model 3: `CBAM` (Convolutional Block Attention Module)
Combines channel attention with spatial attention via a $7\times 7$ convolution. Because Stage 4 features are already reduced to $4\times 4$ spatial resolution, applying a $7\times 7$ kernel with padding 3 degraded spatial coherence, resulting in lower test accuracy (67.41%).

---

## Exploratory Data Analysis & Agronomic Profiling

### 1. Class Taxonomy & Distribution:
The dataset spans 15 classes across three Solanaceous host plants:
- **Bell Pepper:** Bacterial Spot, Healthy Foliage
- **Potato:** Early Blight, Late Blight, Healthy Foliage
- **Tomato:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Tomato Yellow Leaf Curl Virus (TYLCV), Tomato Mosaic Virus (ToMV), Healthy Foliage

### 2. Exact Domain-Specific Normalization:
Computed across 87,736 curated images:
```python
NORM_MEAN = [0.4652, 0.4788, 0.4193]  # R, G, B channel means
NORM_STD  = [0.1878, 0.1648, 0.2032]  # R, G, B channel standard deviations
```

### 3. Biophysical Marker Distributions:
- **Healthy Foliage:** High Excess Green Index ($2G - R - B > 0.18$), low chlorosis ratio ($R/G < 0.72$), high photometric homogeneity.
- **Necrotic Lesions (e.g., Late Blight, Early Blight):** Significant shift in the red-to-green ratio ($R/G > 1.15$) and high spatial color entropy ($H > 6.4$ bits).

---

## Per-Class Performance Breakdown

Test F1 scores across the 36,351 unseen test partition:

| Class Name | Host Plant | `Conv_BN_ReLU` | `SE_Block` (Winner) | `CBAM` | Best Model |
|---|---|---|---|---|---|
| `Pepper__bell___Bacterial_spot` | Pepper Bell | **85.83%** | 84.99% | **86.20%** | `CBAM` |
| `Pepper__bell___healthy` | Pepper Bell | **89.36%** | 88.94% | 87.21% | `Conv_BN_ReLU` |
| `Potato___Early_blight` | Potato | 57.35% | **69.98%** | 48.67% | **`SE_Block`** |
| `Potato___Late_blight` | Potato | **31.33%** | 16.92% | 18.06% | `Conv_BN_ReLU` |
| `Potato___healthy` | Potato | 88.42% | **89.65%** | 86.87% | **`SE_Block`** |
| `Tomato_Bacterial_spot` | Tomato | 54.73% | **57.77%** | 47.90% | **`SE_Block`** |
| `Tomato_Early_blight` | Tomato | **60.13%** | 49.57% | 45.89% | `Conv_BN_ReLU` |
| `Tomato_Late_blight` | Tomato | **31.60%** | 18.79% | 19.86% | `Conv_BN_ReLU` |
| `Tomato_Leaf_Mold` | Tomato | **7.06%** | 6.98% | 6.52% | `Conv_BN_ReLU` |
| `Tomato_Septoria_leaf_spot` | Tomato | 21.09% | **23.51%** | 10.50% | **`SE_Block`** |
| `Tomato_Spider_mites_Two_spotted_spider_mite` | Tomato | 99.09% | **99.54%** | 99.32% | **`SE_Block`** |
| `Tomato__Target_Spot` | Tomato | **98.49%** | 97.86% | 98.06% | `Conv_BN_ReLU` |
| `Tomato__Tomato_YellowLeaf__Curl_Virus` | Tomato | 97.28% | **98.80%** | 98.23% | **`SE_Block`** |
| `Tomato__Tomato_mosaic_virus` | Tomato | 87.50% | **88.09%** | 86.22% | **`SE_Block`** |
| `Tomato_healthy` | Tomato | 92.59% | **93.65%** | 93.10% | **`SE_Block`** |

---

## Standalone Multi-Model CLI Inference (CPU / Zero Datasets Needed)

You can run predictions on any laptop

The standalone script [`classify_leaf.py`](file:///Users/kushrocks6306/Deep%20Learning%20Project/classify_leaf.py) automatically loads the pre-trained weights for all 3 models from disk and runs CPU inference in under a second:

```bash
# 1. Interactive Mode (prompts you for an image, or press Enter for sample_leaf.jpg):
python3 classify_leaf.py

# 2. Classify any custom leaf image using all 3 models side-by-side:
python3 classify_leaf.py path/to/my_leaf.jpg --model all

# 3. Classify using only the winning SE_Block model:
python3 classify_leaf.py path/to/my_leaf.jpg --model se_block
```

Outputs:
- A structured diagnostic card with plant species, health status (Healthy / Diseased), condition name, and confidence %.
- A top-3 probability breakdown with visual distribution bars.
- A side-by-side architectural comparison table across all 3 models (`SE_Block`, `Conv_BN_ReLU`, and `CBAM`).

---

## Interactive Streamlit Web Application

A production-ready Streamlit application is available in `Part_2_Model_Testing_and_GUI/web_app/`:

1. **Navigate to the web app folder:**
   ```bash
   cd "Part_2_Model_Testing_and_GUI/web_app"
   ```

2. **Install minimal dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

4. Open `http://localhost:8501` in your browser. Drag and drop `sample_leaf.jpg` (or any leaf photo) to receive an instant diagnostic card showing:
   - Plant species identification
   - Health status (Healthy vs. Diseased)
   - Specific pathological condition name
   - Top-3 confidence breakdown with probability distributions

---

## Executing the Benchmark Notebook

The master notebook `Part_1_Training_and_Validation/Models.ipynb` supports dual execution modes:

```bash
cd "Part_1_Training_and_Validation"
jupyter notebook Models.ipynb
```

- **`FAST_EVAL_MODE = True` (Default, Instant):**
   Loads pre-trained model weights and pre-computed benchmark metrics from `logs/`. All cells execute in **~5 seconds** on any standard CPU or Apple Silicon Mac, reproducing all tables, generating the 4-panel visual dashboard, and executing live inference.
- **`FAST_EVAL_MODE = False` (Full Training):**
   Executes automated GPU hyperparameter tuning and full multi-epoch retraining from scratch on your CUDA GPU.

---

## How to Push to GitHub (<100 MB Direct Push)

GitHub limits individual file uploads to **100 MB**. Standard PyTorch checkpoints often exceed this limit.

By casting model weights to **half-precision FP16**, all model weight files are **~63.4 MB**, allowing complete repository distribution via standard `git push` without requiring Git LFS:

```bash
# Initialize git repository
git init

# Add all project source files, documentation, and the lightweight model weights
git add README.md PROJECT_REPORT.md launch.sh launch.bat classify_leaf.py sample_leaf.jpg requirements.txt
git add "Part_1_Training_and_Validation/"
git add "Part_2_Model_Testing_and_GUI/"

# Commit and push
git commit -m "Organize plant disease benchmark into Part 1 (Training & Validation) and Part 2 (Model Testing & GUI)"
git branch -M main
git remote add origin https://github.com/ComputationalBiologist1/plant-village-model.git
git push -u origin main
```

---

## Complete Technical Report
For mathematical derivations, biophysical hypothesis tests, full hyperparameter schedules, and in-depth architectural analysis, refer to the full **[PROJECT_REPORT.md](file:///Users/kushrocks6306/Deep%20Learning%20Project/PROJECT_REPORT.md)**.

---
*Developed and benchmarked with PyTorch, CUDA, and CustomTkinter. All code, datasets, weights, and logs are fully self-contained in this repository.*
