#!/Library/Frameworks/Python.framework/Versions/3.11/bin/python3
"""
================================================================================
Plant Disease Multi-Model Classifier (Standalone CPU / Laptop Inference Engine)
================================================================================
This script allows you to classify any plant leaf image using the pre-trained
models without requiring any training, validation, or test datasets.

It loads the pre-computed weights for all 3 benchmarked architectures:
  1. SE_Block (ConvNeXt-Tiny + Squeeze-and-Excitation Attention - Selected Champion)
  2. Conv_BN_ReLU (ConvNeXt-Tiny + 3x3 Residual Refinement)
  3. CBAM (ConvNeXt-Tiny + Channel & Spatial Attention Module)

Usage:
  # Interactive mode (prompts for image path, defaults to sample_leaf.jpg):
  python3 classify_leaf.py

  # Direct CLI image path:
  python3 classify_leaf.py path/to/leaf.jpg

  # Run only the winning model:
  python3 classify_leaf.py path/to/leaf.jpg --model se_block

  # Run all 3 models side-by-side:
  python3 classify_leaf.py path/to/leaf.jpg --model all
================================================================================
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from PIL import Image

# Suppress harmless warnings
warnings.filterwarnings("ignore")

# Silent import of torchvision (bypasses NumPy 2.x / PyTorch C-API warnings on macOS/Linux)
devnull = os.open(os.devnull, os.O_WRONLY)
old_stderr = os.dup(2)
try:
    os.dup2(devnull, 2)
    import torchvision.transforms as T
    import torchvision.models as models
finally:
    os.dup2(old_stderr, 2)
    os.close(old_stderr)
    os.close(devnull)

import torch
import torch.nn as nn

# ==============================================================================
# 1. 15-CLASS TAXONOMY & HUMAN-READABLE METADATA MAPPING
# ==============================================================================
CLASS_TAXONOMY = {
    0: {
        "raw_class": "Pepper__bell___Bacterial_spot",
        "plant": "Pepper Bell",
        "health_status": "Diseased",
        "diagnosis": "Bacterial Spot (Xanthomonas campestris)",
        "is_diseased": True,
    },
    1: {
        "raw_class": "Pepper__bell___healthy",
        "plant": "Pepper Bell",
        "health_status": "Healthy",
        "diagnosis": "Healthy Foliage (No Pathology Detected)",
        "is_diseased": False,
    },
    2: {
        "raw_class": "Potato___Early_blight",
        "plant": "Potato",
        "health_status": "Diseased",
        "diagnosis": "Early Blight (Alternaria solani)",
        "is_diseased": True,
    },
    3: {
        "raw_class": "Potato___Late_blight",
        "plant": "Potato",
        "health_status": "Diseased",
        "diagnosis": "Late Blight (Phytophthora infestans)",
        "is_diseased": True,
    },
    4: {
        "raw_class": "Potato___healthy",
        "plant": "Potato",
        "health_status": "Healthy",
        "diagnosis": "Healthy Foliage (No Pathology Detected)",
        "is_diseased": False,
    },
    5: {
        "raw_class": "Tomato_Bacterial_spot",
        "plant": "Tomato",
        "health_status": "Diseased",
        "diagnosis": "Bacterial Spot (Xanthomonas campestris pv. vesicatoria)",
        "is_diseased": True,
    },
    6: {
        "raw_class": "Tomato_Early_blight",
        "plant": "Tomato",
        "health_status": "Diseased",
        "diagnosis": "Early Blight (Alternaria solani)",
        "is_diseased": True,
    },
    7: {
        "raw_class": "Tomato_Late_blight",
        "plant": "Tomato",
        "health_status": "Diseased",
        "diagnosis": "Late Blight (Phytophthora infestans)",
        "is_diseased": True,
    },
    8: {
        "raw_class": "Tomato_Leaf_Mold",
        "plant": "Tomato",
        "health_status": "Diseased",
        "diagnosis": "Leaf Mold (Passalora fulva)",
        "is_diseased": True,
    },
    9: {
        "raw_class": "Tomato_Septoria_leaf_spot",
        "plant": "Tomato",
        "health_status": "Diseased",
        "diagnosis": "Septoria Leaf Spot (Septoria lycopersici)",
        "is_diseased": True,
    },
    10: {
        "raw_class": "Tomato_Spider_mites_Two_spotted_spider_mite",
        "plant": "Tomato",
        "health_status": "Diseased",
        "diagnosis": "Two-Spotted Spider Mite (Tetranychus urticae)",
        "is_diseased": True,
    },
    11: {
        "raw_class": "Tomato__Target_Spot",
        "plant": "Tomato",
        "health_status": "Diseased",
        "diagnosis": "Target Spot (Corynespora cassiicola)",
        "is_diseased": True,
    },
    12: {
        "raw_class": "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "plant": "Tomato",
        "health_status": "Diseased",
        "diagnosis": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "is_diseased": True,
    },
    13: {
        "raw_class": "Tomato__Tomato_mosaic_virus",
        "plant": "Tomato",
        "health_status": "Diseased",
        "diagnosis": "Tomato Mosaic Virus (ToMV)",
        "is_diseased": True,
    },
    14: {
        "raw_class": "Tomato_healthy",
        "plant": "Tomato",
        "health_status": "Healthy",
        "diagnosis": "Healthy Foliage (No Pathology Detected)",
        "is_diseased": False,
    },
}

# Domain-specific PlantVillage normalization constants
NORM_MEAN = [0.4652, 0.4788, 0.4193]
NORM_STD  = [0.1878, 0.1648, 0.2032]
IMG_SIZE  = 128

# ==============================================================================
# 2. NUMPY 2.x SAFE PREPROCESSING PIPELINE
# ==============================================================================
class SafeToTensor(nn.Module):
    """
    Directly converts PIL RGB Image into a PyTorch Float32 Tensor on [0, 1]
    using bytearray buffer slicing, bypassing NumPy C-API compatibility issues.
    """
    def forward(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic
        pic = pic.convert("RGB")
        w, h = pic.size
        byte_tensor = torch.frombuffer(bytearray(pic.tobytes()), dtype=torch.uint8)
        return byte_tensor.view(h, w, 3).permute(2, 0, 1).to(dtype=torch.float32).div(255.0)

def get_preprocessing_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BILINEAR),
        SafeToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

# ==============================================================================
# 3. ARCHITECTURAL DEFINITIONS
# ==============================================================================

# --- MODEL 1: Conv_BN_ReLU (3x3 Residual Refinement) ---
class ConvNeXtTiny_ConvBNReLU(nn.Module):
    def __init__(self, num_classes=15, dropout_p=0.0):
        super().__init__()
        backbone = models.convnext_tiny(weights=None)
        self.features = backbone.features
        self.refinement = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(768, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.refinement(feat) + feat
        pooled = self.pool(feat)
        return self.classifier(pooled)

# --- MODEL 2: SE_Block (Squeeze-and-Excitation Channel Attention - WINNER) ---
class SEBlock(nn.Module):
    def __init__(self, channels=768, reduction=16):
        super().__init__()
        reduced_ch = max(channels // reduction, 16)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_ch, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_ch, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvNeXtTiny_SE(nn.Module):
    def __init__(self, num_classes=15, dropout_p=0.0, reduction=16):
        super().__init__()
        backbone = models.convnext_tiny(weights=None)
        self.features = backbone.features
        self.pre_conv = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.refinement = SEBlock(channels=768, reduction=reduction)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(768, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        refined = self.pre_conv(feat)
        excited = self.refinement(refined) + feat
        pooled = self.pool(excited)
        return self.classifier(pooled)

# --- MODEL 3: CBAM (Channel & Spatial Attention Module) ---
class ChannelGate(nn.Module):
    def __init__(self, gate_channels=768, reduction_ratio=16):
        super().__init__()
        reduced_channels = max(gate_channels // reduction_ratio, 16)
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, gate_channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = torch.mean(x, dim=(2, 3))
        max_pool = torch.amax(x, dim=(2, 3))
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        scale = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.amax(x, dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale

class CBAMBlock(nn.Module):
    def __init__(self, channels=768, reduction_ratio=16):
        super().__init__()
        self.channel_gate = ChannelGate(channels, reduction_ratio)
        self.spatial_gate = SpatialGate()

    def forward(self, x):
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        return x

class ConvNeXtTiny_CBAM(nn.Module):
    def __init__(self, num_classes=15, dropout_p=0.0, reduction=16):
        super().__init__()
        backbone = models.convnext_tiny(weights=None)
        self.features = backbone.features
        self.pre_conv = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.refinement = CBAMBlock(channels=768, reduction_ratio=reduction)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(768, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        refined = self.pre_conv(feat)
        attended = self.refinement(refined) + feat
        pooled = self.pool(attended)
        return self.classifier(pooled)

# ==============================================================================
# 4. PATH LOCATOR UTILITIES
# ==============================================================================
def find_weight_file(filename):
    """Searches multiple plausible directories for model weight files."""
    candidates = [
        Path(filename),
        Path.cwd() / filename,
        Path.cwd() / "Part_2_Model_Testing_and_GUI" / "models" / filename,
        Path.cwd() / "models" / filename,
        Path.cwd() / "final results" / "models" / filename,
        Path.cwd() / "compatible-model" / "models" / filename,
        Path(__file__).resolve().parent / "Part_2_Model_Testing_and_GUI" / "models" / filename,
        Path(__file__).resolve().parent / "models" / filename,
        Path(__file__).resolve().parent / "final results" / "models" / filename,
        Path(__file__).resolve().parent / "compatible-model" / "models" / filename,
        Path(__file__).resolve().parent.parent / "Part_2_Model_Testing_and_GUI" / "models" / filename,
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return None

def find_sample_image():
    """Locates the default sample leaf photo if no image path is passed."""
    candidates = [
        Path("sample_leaf.jpg"),
        Path.cwd() / "sample_leaf.jpg",
        Path.cwd() / "Part_2_Model_Testing_and_GUI" / "sample_leaf.jpg",
        Path.cwd() / "final results" / "web_app" / "sample_leaf.jpg",
        Path.cwd() / "web_app" / "sample_leaf.jpg",
        Path(__file__).resolve().parent / "sample_leaf.jpg",
        Path(__file__).resolve().parent / "Part_2_Model_Testing_and_GUI" / "sample_leaf.jpg",
        Path(__file__).resolve().parent / "final results" / "web_app" / "sample_leaf.jpg",
        Path(__file__).resolve().parent.parent / "Part_2_Model_Testing_and_GUI" / "sample_leaf.jpg",
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return None

# ==============================================================================
# 5. INFERENCE ENGINE CLASS
# ==============================================================================
class LeafClassifierEngine:
    def __init__(self, device=None):
        if device is None:
            # CPU by default for universal laptop compatibility (or CUDA if available)
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.transform = get_preprocessing_transform()
        self.models = {}

        # Architecture configuration registry
        self.model_configs = {
            "se_block": {
                "name": "SE_Block (ConvNeXt-Tiny + Squeeze-and-Excitation) [Champion / Production Selected]",
                "class_fn": ConvNeXtTiny_SE,
                "weight_files": [
                    "weights_se_block_fp16.pth",
                    "plant_disease_model_fp16.pth",
                    "weights_se_block.pth",
                ],
            },
            "conv_bn_relu": {
                "name": "Conv_BN_ReLU (ConvNeXt-Tiny + 3x3 Residual Refinement) [Rank 2]",
                "class_fn": ConvNeXtTiny_ConvBNReLU,
                "weight_files": [
                    "weights_conv_bn_relu_fp16.pth",
                    "weights_conv_bn_relu.pth",
                ],
            },
            "cbam": {
                "name": "CBAM (ConvNeXt-Tiny + Channel & Spatial Attention) [Rank 3]",
                "class_fn": ConvNeXtTiny_CBAM,
                "weight_files": [
                    "weights_cbam_fp16.pth",
                    "weights_cbam.pth",
                ],
            },
        }

    def load_model(self, model_key):
        """Loads a model instance and restores weights onto the target device."""
        if model_key in self.models:
            return self.models[model_key]

        cfg = self.model_configs.get(model_key.lower())
        if not cfg:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(self.model_configs.keys())}")

        weight_path = None
        for wf in cfg["weight_files"]:
            found = find_weight_file(wf)
            if found:
                weight_path = found
                break

        if not weight_path:
            raise FileNotFoundError(
                f"Could not locate pre-trained weights for {cfg['name']}. "
                f"Searched for: {cfg['weight_files']}"
            )

        # Instantiate architecture
        model = cfg["class_fn"](num_classes=15, dropout_p=0.0)

        # Load weights
        checkpoint = torch.load(weight_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = {
                k: v.float() if v.dtype == torch.float16 else v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            state_dict = {
                k: v.float() if v.dtype == torch.float16 else v
                for k, v in checkpoint.items()
            }

        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        self.models[model_key] = model
        return model

    @torch.no_grad()
    def predict_image(self, image_path, model_key="se_block", top_k=3):
        """Classifies a single image with the specified model."""
        model = self.load_model(model_key)
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu()

        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))
        top_probs = top_probs.tolist()
        top_indices = top_indices.tolist()

        best_idx = top_indices[0]
        best_meta = CLASS_TAXONOMY[best_idx]

        candidates = []
        for idx, prob in zip(top_indices, top_probs):
            meta = CLASS_TAXONOMY[idx]
            candidates.append({
                "class_index": idx,
                "plant": meta["plant"],
                "health_status": meta["health_status"],
                "diagnosis": meta["diagnosis"],
                "is_diseased": meta["is_diseased"],
                "confidence_pct": round(prob * 100.0, 2),
                "raw_label": meta["raw_class"],
            })

        return {
            "model_key": model_key,
            "model_name": self.model_configs[model_key]["name"],
            "plant": best_meta["plant"],
            "health_status": best_meta["health_status"],
            "diagnosis": best_meta["diagnosis"],
            "is_diseased": best_meta["is_diseased"],
            "confidence_pct": round(top_probs[0] * 100.0, 2),
            "raw_class": best_meta["raw_class"],
            "top_candidates": candidates,
        }

# ==============================================================================
# 6. FORMATTED CLI OUTPUT REPORT
# ==============================================================================
def print_diagnosis_card(res):
    status_icon = "[DISEASED]" if res["is_diseased"] else "[HEALTHY]"
    print("\n" + "=" * 78)
    print(f"DIAGNOSTIC REPORT: {res['model_name']}")
    print("=" * 78)
    print(f"  Crop Species        : {res['plant']}")
    print(f"  Health Status       : {status_icon} {res['health_status']}")
    print(f"  Pathology Diagnosis : {res['diagnosis']}")
    print(f"  Model Confidence    : {res['confidence_pct']:.2f}%")
    print(f"  Internal Class Tag  : {res['raw_class']}")
    print("-" * 78)
    print("  Top-3 Candidate Probability Breakdown:")
    for i, c in enumerate(res["top_candidates"], start=1):
        bar = "█" * int(c["confidence_pct"] / 5)
        print(f"   [{i}] {c['confidence_pct']:6.2f}% | {bar:<20} | {c['plant']} - {c['diagnosis']}")
    print("=" * 78)

def print_multi_model_comparison(results_list):
    print("\n" + "#" * 78)
    print("MULTI-MODEL SIDE-BY-SIDE ARCHITECTURAL COMPARISON")
    print("#" * 78)
    print(f"{'Model Architecture':<36} | {'Predicted Condition':<28} | {'Confidence':<10}")
    print("-" * 78)
    for r in results_list:
        short_name = r["model_key"].upper()
        diag = r["diagnosis"].split("(")[0].strip()
        conf = f"{r['confidence_pct']:.2f}%"
        print(f"{short_name:<36} | {diag:<28} | {conf:<10}")
    print("#" * 78)

# ==============================================================================
# 7. MAIN INTERACTIVE EXECUTION LOGIC
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Classify any plant leaf photo using pre-trained ConvNeXt models on CPU or GPU."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to the leaf image file (e.g. sample_leaf.jpg)",
    )
    parser.add_argument(
        "--image",
        "-i",
        dest="image_flag",
        default=None,
        help="Path to the leaf image file (flag alias)",
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=["se_block", "conv_bn_relu", "cbam", "all"],
        default="all",
        help="Which model architecture to run: 'se_block' (winner), 'conv_bn_relu', 'cbam', or 'all' (default)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Compute device (defaults to CPU for universal laptop compatibility)",
    )

    args = parser.parse_args()

    # Determine image path
    image_path = args.image_flag if args.image_flag else args.image
    if not image_path:
        default_sample = find_sample_image()
        print("\n" + "=" * 78)
        print("Plant Disease Classifier (Laptop / CPU Standalone Runner)")
        print("=" * 78)
        if default_sample:
            prompt = f"Enter leaf image path [Press Enter to use '{default_sample.name}']: "
        else:
            prompt = "Enter leaf image path: "

        try:
            user_input = input(prompt).strip().strip("'\"")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

        if user_input:
            image_path = Path(user_input)
        elif default_sample:
            image_path = default_sample
        else:
            print("Error: No image path provided.")
            sys.exit(1)
    else:
        image_path = Path(image_path)

    if not image_path.exists():
        print(f"\nError: File not found at: {image_path.resolve()}")
        sys.exit(1)

    print(f"\nAnalyzing Leaf Image: {image_path.resolve()}")

    engine = LeafClassifierEngine(device=args.device)
    print(f"Execution Device     : {engine.device.type.upper()} (Zero Training / Val Set Required)")

    if args.model == "all":
        # Run all three architectures
        results = []
        for m_key in ["se_block", "conv_bn_relu", "cbam"]:
            res = engine.predict_image(image_path, model_key=m_key)
            results.append(res)
            print_diagnosis_card(res)
        print_multi_model_comparison(results)
    else:
        res = engine.predict_image(image_path, model_key=args.model)
        print_diagnosis_card(res)

if __name__ == "__main__":
    main()
