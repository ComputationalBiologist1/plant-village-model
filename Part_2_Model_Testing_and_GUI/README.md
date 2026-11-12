# Part 2: Model Testing, Inference Engines, and GUI Workstation

## Overview
This directory contains everything required to test and deploy the trained botanical pathology models independently. **No datasets need to be downloaded, and no dedicated GPU is required.** The directory includes pre-compiled, optimized FP16 model weights (<65 MB each), a standalone Enterprise Laboratory Information Management System (LIMS) desktop GUI, a headless CLI runner, and a lightweight web application.

---

## Directory Contents

| Component | Path | Description |
| :--- | :--- | :--- |
| **PhytoScan LIMS Desktop GUI** | `plantguard_app.py` | Complete desktop workstation built with CustomTkinter. Features real-time multi-model inference, dynamic radial confidence gauge, Sanzo Wada color palettes, and deterministic clinical care protocols. |
| **macOS / Linux Launcher** | `launch.sh` | Shell launcher with automatic Python virtual environment detection, Tkinter validation, and font configuration. |
| **Windows Launcher** | `launch.bat` | Double-click Windows batch launcher with Per-Monitor V2 High-DPI scaling configuration. |
| **Command-Line Classifier** | `classify_leaf.py` | Standalone headless inference tool. Outputs top-1 pathology, confidence scores, probability distribution across all 15 classes, and deterministic clinical interventions. |
| **CLI Runner Script** | `run_classifier.sh` | One-command shell shortcut to run `classify_leaf.py` on the bundled sample specimen. |
| **Pre-Trained FP16 Weights** | `models/` | Directory containing optimized FP16 checkpoints for all three evaluated architectures: SE-Block (Champion), Conv-BN-ReLU, and CBAM. |
| **Web Testing Interface** | `web_app/` | Alternative Streamlit browser-based testing interface with drag-and-drop file upload. |
| **Sample Specimen** | `sample_leaf.jpg` | Verified botanical pathology specimen image for instant out-of-the-box validation. |
| **Dependencies** | `requirements.txt` | Minimal runtime production dependencies (Torch, Torchvision, Pillow, CustomTkinter). |

---

## 1. PhytoScan Diagnostic Workstation (Desktop GUI)

The PhytoScan Diagnostic Workstation is designed as an industrial laboratory instrument (inspired by Roche Diagnostics and Illumina BaseSpace) rather than an AI chatbot or template dashboard.

### Key Capabilities
- **Deterministic Agronomic Database**: Replaces all generative AI text dependencies with a local, deterministic agronomic clinical database compliant with FRAC (Fungicide Resistance Action Committee) and IRAC (Insecticide Resistance Action Committee) standards.
- **Sanzo Wada Color Palettes**: Switch live between 5 curated palettes derived from Sanzo Wada's historical dictionary of color combinations:
  1. `Shiro-Kuro (Default)`: Matte slate-carbon laboratory surface (`#090C10`) with icy clinical blue accents (`#2F81F7`).
  2. `Botanist Fieldwork`: Deep moss-tinted obsidian (`#0B110D`) with terracotta earth accents (`#D97706`).
  3. `Kuri-Iro (Amber)`: Roasted chestnut dark backdrop (`#0F0D0B`) with warm amber indicators (`#E5A93C`).
  4. `Seiji-Iro (Celadon)`: Deep jade-patina background (`#091211`) with sea-green mineral accents (`#14B8A6`).
  5. `Sumi-E (High Contrast)`: Charcoal sumi ink surface (`#050505`) with crisp clinical white and platinum accents (`#FFFFFF`).
- **Precision Radial Gauge**: Custom anti-aliased canvas gauge rendering real-time animated confidence arcs with tabular monospace readouts.
- **Structured 3-Column Protocol Matrix**: Displays actionable interventions split into:
  - `01 // Containment & Cultural Hygiene`
  - `02 // Chemical & Bio-Fungicidal Interventions (FRAC/IRAC indexed)`
  - `03 // Vector Exclusion & Preventative Rotation`
- **Clinical Report Export**: Export structured diagnostic reports in standard `.JSON` or `.TXT` formats with one click.
- **Cross-Platform OS Engine**: Razor-sharp typography on macOS (`SF Pro`), Windows (`Segoe UI`), and Linux (`Inter` / `DejaVu Sans`). High-DPI aware with cross-platform trackpad/mousewheel scrolling.

### Launching the Desktop Workstation

#### macOS / Linux:
```bash
chmod +x launch.sh
./launch.sh
```

#### Windows:
Double-click `launch.bat` or run in Command Prompt / PowerShell:
```cmd
launch.bat
```

#### Direct Python Execution:
```bash
python3 plantguard_app.py
```

---

## 2. Headless CLI Classifier

For automated pipelines, server environments, or quick terminal verification, use `classify_leaf.py`:

```bash
# Classify the bundled test leaf
python3 classify_leaf.py --image sample_leaf.jpg

# Classify any custom specimen
python3 classify_leaf.py --image /path/to/specimen.jpg

# Quick execution script
chmod +x run_classifier.sh
./run_classifier.sh
```

### CLI Output Format
```
================================================================================
              PHYTOSCAN CLINICAL PATHOLOGY INFERENCE ENGINE
              Deterministic Local Botanical Laboratory System
================================================================================
[SYSTEM] Runtime Hardware: Apple Metal Performance Shaders (mps)
[SYSTEM] Model Loaded: weights_se_block_fp16.pth (ResNet50 + SE-Block Attention)
[SYSTEM] Specimen Image: sample_leaf.jpg (224x224 RGB normalized)
--------------------------------------------------------------------------------
PRIMARY PATHOLOGICAL DIAGNOSIS:
  Taxon / Class   : Tomato___Early_blight
  Pathogen        : Alternaria solani
  Host Plant      : Lycopersicon esculentum (Tomato)
  Confidence      : 99.82% [Nominal High-Confidence Diagnosis]
  Threat Level    : HIGH_EPIDEMIOLOGICAL_RISK
--------------------------------------------------------------------------------
ACTIONABLE AGRONOMIC TREATMENT PROTOCOL:
  [01] CULTURAL HYGIENE:
       Prune and incinerate lower infected foliage showing concentric target spots.
       Transition to sub-canopy drip irrigation; eliminate all overhead watering.
  [02] FUNGICIDAL INTERVENTION:
       Apply chlorothalonil (FRAC M05) preventative barrier immediately.
       Rotate with azoxystrobin (FRAC 11) or difenoconazole (FRAC 3) to prevent resistance.
  [03] ROTATION & VECTOR MANAGEMENT:
       Enforce a 3-year non-solanaceous crop rotation.
       Sanitize field stakes and trellis wires with 10% quaternary ammonium solution.
================================================================================
```

---

## 3. Web Testing Application

To launch the browser-based Streamlit interface:
```bash
pip install -r web_app/requirements.txt
streamlit run web_app/app.py
```

---

## 4. Pre-Compiled Models

All model checkpoints in `models/` are quantized to half-precision (`float16`), keeping individual file sizes strictly under 65 MB (compatible with standard Git push without Git LFS):

1. `weights_se_block_fp16.pth` (63.4 MB) - **Champion Model**: ResNet-50 backbone with Squeeze-and-Excitation channel attention. Test accuracy: 99.39%, Macro F1: 0.9924.
2. `weights_conv_bn_relu_fp16.pth` (63.3 MB) - Baseline architecture without attention heads. Test accuracy: 99.11%, Macro F1: 0.9887.
3. `weights_cbam_fp16.pth` (63.4 MB) - Convolutional Block Attention Module with dual channel and spatial attention. Test accuracy: 99.28%, Macro F1: 0.9909.
4. `plant_disease_model_fp16.pth` (63.4 MB) - Default deployment weights for the Streamlit web engine.

---

## Minimum System Requirements

- **Operating System**: macOS 11+, Windows 10/11, or Linux (Ubuntu 20.04+, Debian 11+, Fedora 36+)
- **Python**: 3.9, 3.10, or 3.11
- **RAM**: 2 GB minimum (4 GB recommended)
- **Disk Space**: ~300 MB for weights and runtime libraries
- **GPU**: Optional (automatically uses NVIDIA CUDA or Apple Silicon MPS if available, falls back to CPU seamlessly).
