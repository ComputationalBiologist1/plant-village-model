import json
import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn

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

class SafeToTensor(nn.Module):
    def forward(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic
        pic = pic.convert("RGB")
        w, h = pic.size
        byte_tensor = torch.frombuffer(bytearray(pic.tobytes()), dtype=torch.uint8)
        return byte_tensor.view(h, w, 3).permute(2, 0, 1).to(dtype=torch.float32).div(255.0)

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

class PlantDiseasePredictor:
    """
    Production-ready plant disease predictor using the winning ConvNeXt-Tiny SE-Block model.
    Loads from either TorchScript, FP32, or FP16 PyTorch checkpoint bundles.
    """
    def __init__(self, bundle_dir="deployment_bundle", use_gpu=True):
        self.bundle_dir = Path(bundle_dir)
        self.device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

        # 1. Load config
        config_path = self.bundle_dir / "deployment_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.img_size = self.config.get("img_size", 128)
        self.classes_meta = self.config.get("classes", {})
        mean = self.config.get("normalization_mean", [0.4652, 0.4788, 0.4193])
        std = self.config.get("normalization_std", [0.1878, 0.1648, 0.2032])

        self.transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            SafeToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        # 2. Try loading TorchScript first; fallback to checkpoint
        ts_path = self.bundle_dir / "plant_disease_model_torchscript.pt"
        ckpt_path = self.bundle_dir / "plant_disease_model_checkpoint.pth"
        fp16_path = self.bundle_dir / "plant_disease_model_fp16.pth"

        if ts_path.exists():
            try:
                self.model = torch.jit.load(str(ts_path), map_location=self.device)
                self.model.eval()
                self.is_torchscript = True
                return
            except Exception:
                pass

        self.is_torchscript = False
        target_ckpt = fp16_path if fp16_path.exists() and not ckpt_path.exists() else ckpt_path
        data = torch.load(target_ckpt, map_location="cpu", weights_only=False)
        self.model = ConvNeXtTiny_SE(num_classes=len(self.classes_meta), dropout_p=0.0)
        
        # Load weights (cast to float if fp16)
        state_dict = {k: v.float() if v.dtype == torch.float16 else v for k, v in data["state_dict"].items()}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

    def predict(self, image_input, top_k=3):
        """
        Accepts PIL Image, file path, or bytes, and returns structured diagnosis.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            import io
            img = Image.open(io.BytesIO(image_input)).convert("RGB")

        if self.is_torchscript:
            # TorchScript model expects [0, 1] raw tensor without normalization
            raw_tensor = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                SafeToTensor(),
            ])(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = self.model(raw_tensor)[0]
        else:
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)[0]

        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))
        top_probs = top_probs.cpu().tolist()
        top_indices = top_indices.cpu().tolist()

        winning_idx = str(top_indices[0])
        winning_meta = self.classes_meta.get(winning_idx, {})

        candidates = []
        for idx, prob in zip(top_indices, top_probs):
            meta = self.classes_meta.get(str(idx), {})
            candidates.append({
                "class_index": int(idx),
                "plant": meta.get("plant", "Unknown"),
                "diagnosis": meta.get("diagnosis", "Unknown"),
                "is_diseased": meta.get("is_diseased", True),
                "confidence_pct": round(prob * 100.0, 2),
                "raw_label": meta.get("raw_class", ""),
            })

        return {
            "predicted_plant": winning_meta.get("plant", "Unknown"),
            "health_status": winning_meta.get("health_status", "Unknown"),
            "diagnosis": winning_meta.get("diagnosis", "Unknown"),
            "confidence_pct": round(top_probs[0] * 100.0, 2),
            "is_diseased": winning_meta.get("is_diseased", True),
            "raw_class_label": winning_meta.get("raw_class", ""),
            "top_k_candidates": candidates,
        }

if __name__ == "__main__":
    predictor = PlantDiseasePredictor()
    candidates = [
        Path(__file__).resolve().parent / "sample_leaf.jpg",
        Path("sample_leaf.jpg"),
        Path("web_app/sample_leaf.jpg"),
    ]
    test_img = next((c for c in candidates if c.exists()), None)
    if test_img.exists():
        res = predictor.predict(test_img)
        print("Test Prediction Successful:")
        print(json.dumps(res, indent=2))
