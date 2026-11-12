import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image, ImageFile

# Crucial for multi-worker child processes handling web datasets with truncated JPEGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SafeToTensor(nn.Module):
    def forward(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic
        pic = pic.convert("RGB")
        w, h = pic.size
        byte_tensor = torch.frombuffer(bytearray(pic.tobytes()), dtype=torch.uint8)
        return byte_tensor.view(h, w, 3).permute(2, 0, 1).to(dtype=torch.float32).div(255.0)

class CleanPlantDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                img_rgb = img.convert("RGB")
        except Exception:
            img_rgb = Image.new("RGB", (128, 128), (0, 0, 0))

        if self.transform:
            img_rgb = self.transform(img_rgb)
        return img_rgb, label
