#!/usr/bin/env python3
"""
================================================================================
PHYTOSCAN · BOTANICAL PATHOLOGY & AGRONOMIC DIAGNOSTIC WORKSTATION
Version: 4.2 (Authentic Desktop Edition)
Architecture: ConvNeXt-Tiny Deep Residual Feature Backbone (15-Class Taxonomy)
Deterministic Subsystem: Integrated Agronomic Care & Treatment Database
Color Palettes: A Dictionary of Color Combinations by Sanzo Wada (1930s)
Supported Platforms: Windows (win32), macOS (darwin), Linux (x11/wayland)
================================================================================
"""

import os
import sys
import json
import time
import copy
import threading
import warnings
import math
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# High-DPI Windows Scaling & Hardware Environment Stabilization
# ------------------------------------------------------------------------------
if sys.platform == "win32":
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# Suppress NumPy 2.x C-API stderr noise on PyTorch module loads
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
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

# ==============================================================================
# 1. CROSS-PLATFORM SYSTEM FONT & ACCELERATION RESOLVER
# ==============================================================================
class OSFontResolver:
    """
    Resolves native system typography across Windows, macOS, and Linux
    to guarantee pixel-perfect rendering on any desktop environment.
    """
    _sys = sys.platform

    @classmethod
    def get_ui(cls, size=11, weight="normal"):
        if cls._sys == "darwin":
            family = "SF Pro Text" if size < 16 else "SF Pro Display"
            return (family, size, weight)
        elif cls._sys == "win32":
            return ("Segoe UI", size, weight)
        else:
            return ("Inter", size, weight)

    @classmethod
    def get_mono(cls, size=10, weight="normal"):
        if cls._sys == "darwin":
            return ("SF Mono", size, weight)
        elif cls._sys == "win32":
            return ("Cascadia Mono", size, weight)
        else:
            return ("JetBrains Mono", size, weight)

    @classmethod
    def get_cursor(cls):
        return "pointinghand" if cls._sys == "darwin" else "hand2"


def detect_compute_device():
    """
    Hardware acceleration auto-detector:
    Checks for NVIDIA CUDA, Apple Silicon Metal (MPS), or Multicore CPU.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return torch.device("cuda:0"), f"NVIDIA CUDA ({gpu_name})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            mps_dev = torch.device("mps")
            _test_t = torch.zeros(1, device=mps_dev)
            return mps_dev, "Apple Silicon Metal (MPS)"
        except Exception:
            pass
    threads = min(8, os.cpu_count() or 4)
    return torch.device("cpu"), f"Host CPU ({threads} Cores)"


# ==============================================================================
# 2. SANZO WADA COLOR COMBINATION PALETTES (和田三造 · 配色の事典)
# ==============================================================================
SANZO_WADA_PALETTES = {
    "wada_182": {
        "id": "wada_182",
        "name": "Celadon & Sandalwood",
        "japanese": "青磁と白檀",
        "wada_num": 182,
        "menu_label": "No. 182 · Celadon & Sandalwood (青磁と白檀)",
        # Canvas / Surface
        "bg_root": "#121715",
        "bg_panel": "#1A2320",
        "bg_card": "#212C28",
        "bg_card_sub": "#17201D",
        "bg_input": "#101614",
        "border_hairline": "#2C3E37",
        "border_focus": "#5C8A79",
        "border_subtle": "#1B2723",
        # Accents
        "accent": "#5C8A79",          # Celadon / Seiji green
        "accent_hover": "#71A492",
        "accent_dim": "#1C2C26",
        "secondary": "#C49A6C",       # Sandalwood / Byakudan
        "secondary_hover": "#D4AC80",
        "tertiary": "#8EA37A",        # Muted leaf
        # Status
        "status_nominal": "#5CA866",
        "status_nominal_bg": "#14281B",
        "status_nominal_txt": "#6EC87A",
        "status_warning": "#D4A373",
        "status_warning_bg": "#2B2014",
        "status_warning_txt": "#E5B889",
        "status_critical": "#C85A5A",
        "status_critical_bg": "#2C1414",
        "status_critical_txt": "#E27373",
        # Typography
        "text_primary": "#E8ECE9",
        "text_secondary": "#A3B5AE",
        "text_muted": "#5D736A",
        "bar_track": "#141C19",
    },
    "wada_156": {
        "id": "wada_156",
        "name": "Indigo & Persimmon",
        "japanese": "藍と柿",
        "wada_num": 156,
        "menu_label": "No. 156 · Indigo & Persimmon (藍と柿)",
        "bg_root": "#0F141A",
        "bg_panel": "#161F28",
        "bg_card": "#1D2936",
        "bg_card_sub": "#141C25",
        "bg_input": "#0D1217",
        "border_hairline": "#2B3C4E",
        "border_focus": "#D96B43",
        "border_subtle": "#1B2733",
        "accent": "#D96B43",          # Persimmon / Kaki
        "accent_hover": "#E87D56",
        "accent_dim": "#2C1A14",
        "secondary": "#4E88C7",       # Pure Indigo / Ai
        "secondary_hover": "#629BD9",
        "tertiary": "#CCA352",        # Gold straw
        "status_nominal": "#4EAA77",
        "status_nominal_bg": "#12261B",
        "status_nominal_txt": "#62C78F",
        "status_warning": "#E09F3E",
        "status_warning_bg": "#2B2012",
        "status_warning_txt": "#F0B256",
        "status_critical": "#D9534F",
        "status_critical_bg": "#2B1515",
        "status_critical_txt": "#ED6E6A",
        "text_primary": "#F0F4F8",
        "text_secondary": "#9EB5CC",
        "text_muted": "#59738B",
        "bar_track": "#111822",
    },
    "wada_24": {
        "id": "wada_24",
        "name": "Verdigris & Muted Coral",
        "japanese": "緑青と珊瑚",
        "wada_num": 24,
        "menu_label": "No. 24 · Verdigris & Muted Coral (緑青と珊瑚)",
        "bg_root": "#0E1614",
        "bg_panel": "#15221F",
        "bg_card": "#1E302C",
        "bg_card_sub": "#131E1B",
        "bg_input": "#0C1311",
        "border_hairline": "#2A443E",
        "border_focus": "#389B84",
        "border_subtle": "#1A2C28",
        "accent": "#389B84",          # Verdigris / Rokushō
        "accent_hover": "#4BB59B",
        "accent_dim": "#152924",
        "secondary": "#DF6D63",       # Muted Coral / Sango
        "secondary_hover": "#EC7E74",
        "tertiary": "#8AA39B",        # Sage mineral
        "status_nominal": "#48A874",
        "status_nominal_bg": "#11261B",
        "status_nominal_txt": "#5EC68E",
        "status_warning": "#DDA15E",
        "status_warning_bg": "#2C2013",
        "status_warning_txt": "#EDB574",
        "status_critical": "#D64545",
        "status_critical_bg": "#2B1313",
        "status_critical_txt": "#E86161",
        "text_primary": "#EBF2EF",
        "text_secondary": "#94B5AB",
        "text_muted": "#537368",
        "bar_track": "#121C1A",
    },
    "wada_1": {
        "id": "wada_1",
        "name": "Roasted Tea & Mustard",
        "japanese": "焙じ茶と芥子",
        "wada_num": 1,
        "menu_label": "No. 1 · Roasted Tea & Mustard (焙じ茶と芥子)",
        "bg_root": "#161310",
        "bg_panel": "#221C18",
        "bg_card": "#2D2520",
        "bg_card_sub": "#1C1713",
        "bg_input": "#120F0D",
        "border_hairline": "#42362E",
        "border_focus": "#D4A359",
        "border_subtle": "#2B231D",
        "accent": "#D4A359",          # Karashi mustard gold
        "accent_hover": "#E5B66C",
        "accent_dim": "#2C2112",
        "secondary": "#A3704C",       # Roasted tea / Hōjicha
        "secondary_hover": "#B8825D",
        "tertiary": "#829168",        # Olive foliage
        "status_nominal": "#6A9A5B",
        "status_nominal_bg": "#172615",
        "status_nominal_txt": "#7FB76E",
        "status_warning": "#D89B48",
        "status_warning_bg": "#2B1F10",
        "status_warning_txt": "#EBB05E",
        "status_critical": "#C04A3E",
        "status_critical_bg": "#2B1412",
        "status_critical_txt": "#D9665A",
        "text_primary": "#F7F2EB",
        "text_secondary": "#C2B2A3",
        "text_muted": "#77675A",
        "bar_track": "#181412",
    },
    "wada_68": {
        "id": "wada_68",
        "name": "Wisteria & Raw Umber",
        "japanese": "藤と生木",
        "wada_num": 68,
        "menu_label": "No. 68 · Wisteria & Raw Umber (藤と生木)",
        "bg_root": "#131116",
        "bg_panel": "#1B1722",
        "bg_card": "#262030",
        "bg_card_sub": "#17131F",
        "bg_input": "#0F0D12",
        "border_hairline": "#3B314B",
        "border_focus": "#987AB8",
        "border_subtle": "#261F30",
        "accent": "#987AB8",          # Kyoto wisteria / Fuji
        "accent_hover": "#AE90CF",
        "accent_dim": "#241B2E",
        "secondary": "#C29665",       # Raw timber / Namaki
        "secondary_hover": "#D4A977",
        "tertiary": "#7A9C96",        # Slate jade
        "status_nominal": "#5AA472",
        "status_nominal_bg": "#132619",
        "status_nominal_txt": "#6EC489",
        "status_warning": "#D89F53",
        "status_warning_bg": "#2B2011",
        "status_warning_txt": "#EBB469",
        "status_critical": "#C94B5C",
        "status_critical_bg": "#2B1317",
        "status_critical_txt": "#E06677",
        "text_primary": "#F3EFF7",
        "text_secondary": "#B2A4C2",
        "text_muted": "#6A5C7A",
        "bar_track": "#16131B",
    },
    "wada_322": {
        "id": "wada_322",
        "name": "Dawn Mist & Camellia",
        "japanese": "浅葱と椿",
        "wada_num": 322,
        "menu_label": "No. 322 · Dawn Mist & Camellia (浅葱と椿)",
        "bg_root": "#0E1417",
        "bg_panel": "#141D23",
        "bg_card": "#1B272F",
        "bg_card_sub": "#11181D",
        "bg_input": "#0A0E10",
        "border_hairline": "#2A3D49",
        "border_focus": "#569BA8",
        "border_subtle": "#19252D",
        "accent": "#569BA8",          # Asagi mist cyan
        "accent_hover": "#6CB3C2",
        "accent_dim": "#14252B",
        "secondary": "#D44A59",       # Winter camellia / Tsubaki
        "secondary_hover": "#E55C6B",
        "tertiary": "#E2B15B",        # Morning glow
        "status_nominal": "#4BA575",
        "status_nominal_bg": "#10261A",
        "status_nominal_txt": "#5FC28C",
        "status_warning": "#D99C4B",
        "status_warning_bg": "#2B1F11",
        "status_warning_txt": "#EDB160",
        "status_critical": "#D44A59",
        "status_critical_bg": "#2B1216",
        "status_critical_txt": "#E86473",
        "text_primary": "#EDF5F7",
        "text_secondary": "#97B4C2",
        "text_muted": "#516F7D",
        "bar_track": "#121A1F",
    },
    "wada_270": {
        "id": "wada_270",
        "name": "Sumi Ink & Cinnabar",
        "japanese": "墨と朱色",
        "wada_num": 270,
        "menu_label": "No. 270 · Sumi Ink & Cinnabar (墨と朱色)",
        "bg_root": "#111112",
        "bg_panel": "#19191B",
        "bg_card": "#222226",
        "bg_card_sub": "#151517",
        "bg_input": "#0C0C0D",
        "border_hairline": "#35353C",
        "border_focus": "#E04738",
        "border_subtle": "#212126",
        "accent": "#E04738",          # Cinnabar seal red / Shu
        "accent_hover": "#F05C4D",
        "accent_dim": "#2C1311",
        "secondary": "#A0A0AA",       # Polished inkstone grey
        "secondary_hover": "#B4B4BE",
        "tertiary": "#C89552",        # Gold lacquer
        "status_nominal": "#48A869",
        "status_nominal_bg": "#102517",
        "status_nominal_txt": "#5EC482",
        "status_warning": "#DCA03E",
        "status_warning_bg": "#2B1F0F",
        "status_warning_txt": "#EDB553",
        "status_critical": "#E04738",
        "status_critical_bg": "#2B1210",
        "status_critical_txt": "#F06152",
        "text_primary": "#F4F4F5",
        "text_secondary": "#9E9EA8",
        "text_muted": "#5B5B65",
        "bar_track": "#161618",
    },
    "wada_94": {
        "id": "wada_94",
        "name": "Pine Needle & Smoked Bamboo",
        "japanese": "松葉と煤竹",
        "wada_num": 94,
        "menu_label": "No. 94 · Pine Needle & Smoked Bamboo (松葉と煤竹)",
        "bg_root": "#101511",
        "bg_panel": "#161F18",
        "bg_card": "#1F2B21",
        "bg_card_sub": "#131A14",
        "bg_input": "#0C100D",
        "border_hairline": "#2E3F32",
        "border_focus": "#6E9C68",
        "border_subtle": "#1C271E",
        "accent": "#6E9C68",          # Pine needle / Matsuba
        "accent_hover": "#85B57E",
        "accent_dim": "#172718",
        "secondary": "#A68A5E",       # Smoked bamboo / Susudake
        "secondary_hover": "#BA9D6E",
        "tertiary": "#8A7E9B",        # Shadowed wisteria
        "status_nominal": "#5BAC64",
        "status_nominal_bg": "#132616",
        "status_nominal_txt": "#6EC677",
        "status_warning": "#CCA046",
        "status_warning_bg": "#292010",
        "status_warning_txt": "#E0B359",
        "status_critical": "#C54E4E",
        "status_critical_bg": "#291313",
        "status_critical_txt": "#DB6A6A",
        "text_primary": "#EFF5F0",
        "text_secondary": "#A1B5A3",
        "text_muted": "#586F5B",
        "bar_track": "#131C15",
    },
}

# Active working theme dictionary (dynamically modified on palette switch)
THEME = copy.deepcopy(SANZO_WADA_PALETTES["wada_182"])

# ==============================================================================
# 3. 15-CLASS TAXONOMY & ETIOLOGICAL REPOSITORY
# ==============================================================================
CLASS_TAXONOMY = {
    0:  {"raw": "Pepper__bell___Bacterial_spot", "plant": "Pepper Bell", "status": "Diseased", "diag": "Bacterial Spot", "pathogen": "Xanthomonas campestris pv. vesicatoria", "diseased": True},
    1:  {"raw": "Pepper__bell___healthy", "plant": "Pepper Bell", "status": "Healthy", "diag": "Healthy Foliage", "pathogen": "Physiologically Normal (Zero Lesions)", "diseased": False},
    2:  {"raw": "Potato___Early_blight", "plant": "Potato", "status": "Diseased", "diag": "Early Blight", "pathogen": "Alternaria solani (Ascomycota)", "diseased": True},
    3:  {"raw": "Potato___Late_blight", "plant": "Potato", "status": "Diseased", "diag": "Late Blight", "pathogen": "Phytophthora infestans (Oomycota)", "diseased": True},
    4:  {"raw": "Potato___healthy", "plant": "Potato", "status": "Healthy", "diag": "Healthy Foliage", "pathogen": "Physiologically Normal (Zero Lesions)", "diseased": False},
    5:  {"raw": "Tomato_Bacterial_spot", "plant": "Tomato", "status": "Diseased", "diag": "Bacterial Spot", "pathogen": "Xanthomonas campestris pv. vesicatoria", "diseased": True},
    6:  {"raw": "Tomato_Early_blight", "plant": "Tomato", "status": "Diseased", "diag": "Early Blight", "pathogen": "Alternaria solani (Ascomycota)", "diseased": True},
    7:  {"raw": "Tomato_Late_blight", "plant": "Tomato", "status": "Diseased", "diag": "Late Blight", "pathogen": "Phytophthora infestans (Oomycota)", "diseased": True},
    8:  {"raw": "Tomato_Leaf_Mold", "plant": "Tomato", "status": "Diseased", "diag": "Leaf Mold", "pathogen": "Passalora fulva (Cladosporium fulvum)", "diseased": True},
    9:  {"raw": "Tomato_Septoria_leaf_spot", "plant": "Tomato", "status": "Diseased", "diag": "Septoria Leaf Spot", "pathogen": "Septoria lycopersici (Mycosphaerellaceae)", "diseased": True},
    10: {"raw": "Tomato_Spider_mites_Two_spotted_spider_mite", "plant": "Tomato", "status": "Diseased", "diag": "Two-Spotted Spider Mite", "pathogen": "Tetranychus urticae (Acari: Tetranychidae)", "diseased": True},
    11: {"raw": "Tomato__Target_Spot", "plant": "Tomato", "status": "Diseased", "diag": "Target Spot", "pathogen": "Corynespora cassiicola (Pleosporales)", "diseased": True},
    12: {"raw": "Tomato__Tomato_YellowLeaf__Curl_Virus", "plant": "Tomato", "status": "Diseased", "diag": "Yellow Leaf Curl Virus", "pathogen": "Begomovirus TYLCV (Vector: Bemisia tabaci)", "diseased": True},
    13: {"raw": "Tomato__Tomato_mosaic_virus", "plant": "Tomato", "status": "Diseased", "diag": "Tomato Mosaic Virus", "pathogen": "Tobamovirus ToMV (Mechanical Transmission)", "diseased": True},
    14: {"raw": "Tomato_healthy", "plant": "Tomato", "status": "Healthy", "diag": "Healthy Foliage", "pathogen": "Physiologically Normal (Zero Lesions)", "diseased": False},
}

NORM_MEAN = [0.4652, 0.4788, 0.4193]
NORM_STD  = [0.1878, 0.1648, 0.2032]
IMG_SIZE  = 128

BENCHMARK_PROFILES = {
    "se_block": {
        "name": "Model 1: ConvNeXt-SE (Primary Baseline)",
        "short_name": "ConvNeXt-SE (Primary)",
        "acc": 72.40,
        "sensitivity": 82.75,
        "precision": 68.37,
        "loss": 1.3223,
        "gap": 3.53,
        "params": "33.21M",
        "color": THEME["accent"],
    },
    "conv_bn_relu": {
        "name": "Model 2: ConvNeXt-Residual (3x3 Refinement)",
        "short_name": "ConvNeXt-Residual",
        "acc": 68.89,
        "sensitivity": 81.32,
        "precision": 71.06,
        "loss": 1.4339,
        "gap": 13.95,
        "params": "33.14M",
        "color": THEME["secondary"],
    },
    "cbam": {
        "name": "Model 3: ConvNeXt-CBAM (Dual Attention)",
        "short_name": "ConvNeXt-CBAM",
        "acc": 67.41,
        "sensitivity": 81.50,
        "precision": 67.79,
        "loss": 1.4054,
        "gap": 8.84,
        "params": "33.21M",
        "color": THEME["tertiary"],
    },
}

# ==============================================================================
# 4. DETERMINISTIC CLINICAL AGRONOMIC DATABASE
# ==============================================================================
CLINICAL_TREATMENT_PROTOCOLS = {
    0: {
        "condition": "Pepper Bell: Bacterial Spot (Xanthomonas campestris pv. vesicatoria)",
        "pathogen": "Xanthomonas campestris pv. vesicatoria (Gram-Negative Plant Bacterium)",
        "threat_level": "ELEVATED",
        "threat_type": "WARNING",
        "cultural": [
            "Terminate overhead sprinkler irrigation immediately; transition strictly to sub-canopy drip lines to maintain zero foliar surface wetness.",
            "Enforce strict tool sanitation: sterilize harvesting shears in 10% quaternary ammonium or 70% isopropanol solution every 30 minutes.",
            "Establish an immediate 2-to-3 year crop rotation away from all Solanaceous species (Capsicum, Solanum, Lycopersicon).",
            "Deep-incorporate crop residues to >= 25 cm depth post-harvest to accelerate microbial decomposition of bacterial overwintering tissue."
        ],
        "chemical": [
            "Foliar Bactericide: Fixed Copper Hydroxide (50% WP) @ 2.0-2.5 kg/ha tank-mixed with Mancozeb (75% WG) @ 1.8 kg/ha at 7-day intervals.",
            "Systemic Acquired Resistance Inducer: Acibenzolar-S-methyl (Actigard 50WG) @ 25-35 g/ha to stimulate endogenous phytoalexin pathways.",
            "Bio-Control Agent: Bacillus amyloliquefaciens strain D747 @ 2.0 L/ha applied preventatively during warm overcast intervals."
        ],
        "preventative": [
            "Source exclusively certified pathogen-indexed seed stock; mandate hot-water seed thermal therapy (50°C for exactly 25 minutes).",
            "Select commercial cultivars possessing multi-allelic resistance genes (Bs1, Bs2, Bs3 hypersensitivity loci)."
        ]
    },
    1: {
        "condition": "Pepper Bell: Physiological Equilibrium (Healthy Foliage)",
        "pathogen": "Physiologically Normal (Zero Microbial Lesions)",
        "threat_level": "NOMINAL",
        "threat_type": "NOMINAL",
        "cultural": [
            "Maintain current automated irrigation scheduling; regulate soil matric potential between -25 and -40 kPa.",
            "Conduct routine bi-weekly petiole sap analysis to sustain optimal nitrate (NO3-N) and potassium (K) vegetative balances."
        ],
        "chemical": [
            "Zero curative chemical intervention required.",
            "Prophylactic Foliar Fortification: Apply soluble Potassium Silicate (20% SiO2) @ 2.0 mL/L to polymerize foliar epidermal cuticles."
        ],
        "preventative": [
            "Deploy high-contrast yellow sticky monitoring cards (1 card / 150 m²) for early detection of migratory Thrips and Aphids.",
            "Execute rootzone inoculation of beneficial mycorrhizae (Glomus intraradices) to maximize nutrient uptake efficiency."
        ]
    },
    2: {
        "condition": "Potato: Early Blight (Alternaria solani)",
        "pathogen": "Alternaria solani (Pleosporalean Ascomycete)",
        "threat_level": "MODERATE",
        "threat_type": "WARNING",
        "cultural": [
            "Remove and incinerate senescent lower-canopy foliage exhibiting concentric zonate necrotic lesions.",
            "Correct soil nitrogen deficits; nitrogen-deficient tissues undergo accelerated senescence, acting as primary infection foci.",
            "Avoid deficit irrigation during active tuber initiation and bulking to prevent physiological canopy stress."
        ],
        "chemical": [
            "Protectant Schedule: Chlorothalonil 720 SC @ 1.8-2.2 L/ha OR Mancozeb 75 WG @ 2.0-2.5 kg/ha on a rigid 7-to-10 day cycle.",
            "Systemic Curative Therapy: Difenoconazole (FRAC 3) @ 250 mL/ha combined with Azoxystrobin (FRAC 11) @ 400 mL/ha, or Boscalid (FRAC 7).",
            "Anti-Resistance Protocol: Rotate chemistries strictly across FRAC 3, 7, and 11 to avoid target-site CYP51 / SdhB mutations."
        ],
        "preventative": [
            "Mandate a 3-year non-solanaceous rotational sequence with Poaceae crops (wheat, barley, forage sorghum).",
            "Maintain adequate furrow ridge volume to protect developing tubers from conidial spore wash-down."
        ]
    },
    3: {
        "condition": "Potato: Late Blight (Phytophthora infestans)",
        "pathogen": "Phytophthora infestans (Peronosporalean Oomycete)",
        "threat_level": "CRITICAL EMERGENCY",
        "threat_type": "CRITICAL",
        "cultural": [
            "Immediate physical destruction: Bag infected focal vines in situ before removal; bury to a depth >= 1.0 m away from production areas.",
            "Terminate overhead pivot irrigation; restrict all tractor passes to dry daylight periods to eliminate spore tracking.",
            "Perform chemical vine desiccation (Diquat dibromide @ 2.0 L/ha) 14 days before harvest to eliminate living foliar sporangia."
        ],
        "chemical": [
            "Emergency Systemic Oomyceticide: Cyazofamid (FRAC 21) @ 80-100 mL/ha OR Fluopicolide (FRAC 43) + Propamocarb @ 1.5 L/ha on 5-day intervals.",
            "Translaminar Curative Chemistry: Mandipropamid (FRAC 40) @ 600 mL/ha tank-mixed with protectant Mancozeb (1.5 kg/ha).",
            "Anti-Resistance Warning: Assume resistance to Mefenoxam/Metalaxyl (FRAC 4) in US-23 and Blue-13 clonal lineages; avoid solitary use."
        ],
        "preventative": [
            "Eradicate all cull piles, volunteer potato regrowth, and Solanum nigrum nightshade hosts within a 5 km perimeter.",
            "Inspect all incoming seed tubers pre-sprouting at 10-15°C under forced ventilation for sunken copper-brown dry rot lesions."
        ]
    },
    4: {
        "condition": "Potato: Physiological Equilibrium (Healthy Foliage)",
        "pathogen": "Physiologically Normal (Zero Microbial Lesions)",
        "threat_level": "NOMINAL",
        "threat_type": "NOMINAL",
        "cultural": [
            "Maintain canopy hilling to protect developing tubers from solar greening and late blight spore washdown.",
            "Verify soil drainage efficiency; waterlogging promotes bacterial soft rot (Pectobacterium carotovorum) colonization."
        ],
        "chemical": [
            "Zero curative chemical intervention required.",
            "Nutritional Support: Foliar application of Zinc-Boron micronutrient complex @ 1.0 kg/ha to maintain apical cell division during tuber initiation."
        ],
        "preventative": [
            "Monitor regional Late Blight Decision Support Systems (DSS) for Blight Units and Smith Periods.",
            "Install spore traps along prevailing wind perimeters to detect aerial sporangial influx."
        ]
    },
    5: {
        "condition": "Tomato: Bacterial Spot (Xanthomonas campestris pv. vesicatoria)",
        "pathogen": "Xanthomonas campestris pv. vesicatoria (Gram-Negative Plant Bacterium)",
        "threat_level": "ELEVATED",
        "threat_type": "WARNING",
        "cultural": [
            "Prohibit field cultivation or harvesting when foliage carries dew or precipitation.",
            "Sterilize greenhouse stakes, wires, and pruning shears in 20% sodium hypochlorite solution between crop cycles.",
            "Establish tall non-host barrier windbreaks (e.g., Sorghum-Sudangrass) to attenuate wind-driven aerosol transmission."
        ],
        "chemical": [
            "Protective Bactericide: Basic Copper Sulfate (53% metallic copper equivalent) @ 2.0 kg/ha tank-mixed with Mancozeb @ 1.5 kg/ha.",
            "Biological Bacteriophage Therapy: AgriPhage-C50 (specific lytic bacteriophages targeting Xanthomonas strains) applied at dusk to evade UV degradation.",
            "Systemic Defense Activator: Prohexadione-calcium @ 100 mg/L to modulate vegetative growth and cell wall resistance."
        ],
        "preventative": [
            "Procure certified pathogen-free hybrid seeds; perform hot-water thermal therapy (50°C for 25 min) on uncertified seed lots.",
            "Maintain greenhouse relative humidity < 80% with automated ventilation and thermal pulse heating."
        ]
    },
    6: {
        "condition": "Tomato: Early Blight (Alternaria solani)",
        "pathogen": "Alternaria solani (Pleosporalean Ascomycete)",
        "threat_level": "MODERATE",
        "threat_type": "WARNING",
        "cultural": [
            "Prune off all indeterminate lateral suckers and lower foliage up to 30 cm from ground level to promote laminar airflow.",
            "Apply polyethylene or organic straw mulch across all beds to suppress rain-splash of soil-borne conidia.",
            "Irrigate strictly via pressure-compensating drip tubing beneath black mulch."
        ],
        "chemical": [
            "Foliar Protectant: Chlorothalonil (FRAC M05) @ 2.0 L/ha or Metiram (FRAC M03) @ 2.2 kg/ha.",
            "Systemic Fungicide Rotation: Fluxapyroxad + Pyraclostrobin (Priaxor, FRAC 7 + 11) @ 300 mL/ha alternating with Difenoconazole (FRAC 3) @ 250 mL/ha.",
            "Spray Volume: Calibrate ground rig to minimum 400 L/ha water volume to achieve thorough lower-leaf under-surface coverage."
        ],
        "preventative": [
            "Maintain minimum 3-year field rotation away from solanaceous and brassicaceous species.",
            "Incorporate crop residues immediately post-final harvest via disc-plow incorporation."
        ]
    },
    7: {
        "condition": "Tomato: Late Blight (Phytophthora infestans)",
        "pathogen": "Phytophthora infestans (Peronosporalean Oomycete)",
        "threat_level": "CRITICAL EMERGENCY",
        "threat_type": "CRITICAL",
        "cultural": [
            "Execute immediate quarantine of affected greenhouse bays or field sections.",
            "Remove and hermetically seal all blighted plants in polyethylene bags before removal from the field; do not compost.",
            "Drop relative humidity in protected structures to < 70% by simultaneous exhaust venting and heating."
        ],
        "chemical": [
            "Translaminar / Systemic Oomyceticide: Fluopicolide (Presidio, FRAC 43) @ 250 mL/ha combined with Chlorothalonil (1.5 L/ha).",
            "Alternative FRAC 40/21 Option: Dimethomorph @ 400 g/ha OR Cyazofamid (Ranman, FRAC 21) @ 100 mL/ha at 5-day spray intervals under active disease pressure.",
            "Phyto-toxicity Safeguard: Avoid organosilicone surfactants when tank-mixing copper with systemic fungicides during high temperature intervals."
        ],
        "preventative": [
            "Plant resistant indeterminate tomato hybrids carrying Ph-2 and Ph-3 dual resistance genes (e.g., 'Defiant PhR', 'Mountain Merit').",
            "Subscribe to regional plant pathology extension alerts for aerobiological sporangial forecasts."
        ]
    },
    8: {
        "condition": "Tomato: Leaf Mold (Passalora fulva / Cladosporium fulvum)",
        "pathogen": "Passalora fulva (Cladosporium fulvum)",
        "threat_level": "MODERATE",
        "threat_type": "WARNING",
        "cultural": [
            "Dehumidification Priority: Regulate greenhouse climate to sustain ambient relative humidity strictly below 85%.",
            "Improve plant spacing to minimum 50 cm between stems in twin-row arrangements to enhance air velocity across leaf surfaces.",
            "Increase night-time heating setpoints to avoid condensation formation at leaf boundary layers."
        ],
        "chemical": [
            "Protectant Application: Copper Octanoate (Copper Soap) @ 3.0 L/ha applied at first appearance of pale chlorotic upper-surface blotches.",
            "Curative Systemic Fungicide: Difenoconazole (FRAC 3) @ 300 mL/ha or Triflumizole (Procure 480SC, FRAC 3) @ 400 mL/ha.",
            "Bio-fungicide: Bacillus subtilis strain QST 713 (Serenade ASO) @ 4.0 L/ha applied weekly under moderate humidity."
        ],
        "preventative": [
            "Select greenhouse tomato cultivars integrated with Cf-2, Cf-4, Cf-5, or Cf-9 multi-allelic resistance genes.",
            "Perform annual thermal dry-heat greenhouse sterilization (40°C for 24 hours) between cycles."
        ]
    },
    9: {
        "condition": "Tomato: Septoria Leaf Spot (Septoria lycopersici)",
        "pathogen": "Septoria lycopersici (Mycosphaerellaceae)",
        "threat_level": "MODERATE",
        "threat_type": "WARNING",
        "cultural": [
            "Immediately strip and discard lower leaves exhibiting circular necrotic spots with greyish-white centres and dark pycnidia.",
            "Maintain dry foliar canopies; restrict automated pruning operations to low relative humidity hours.",
            "Suppress all surrounding solanaceous weeds (Horse Nettle, Black Nightshade) harboring overwintering perithecia."
        ],
        "chemical": [
            "Standard Protectant Schedule: Chlorothalonil 720 SC @ 2.0 L/ha or Copper Hydroxide @ 2.0 kg/ha beginning at seedling establishment.",
            "Curative Triazole Spray: Tebuconazole (FRAC 3) @ 250 mL/ha or Pyraclostrobin (FRAC 11) @ 400 mL/ha on 10-day intervals.",
            "Application Standard: Utilize hollow-cone nozzles with minimum 3.0 bar operational pressure to penetrate dense canopies."
        ],
        "preventative": [
            "Mandate 3-year non-solanaceous crop rotations.",
            "Deep-plow crop residues to a minimum depth of 20 cm immediately following terminal harvest."
        ]
    },
    10: {
        "condition": "Tomato: Two-Spotted Spider Mite (Tetranychus urticae)",
        "pathogen": "Tetranychus urticae (Acari: Tetranychidae)",
        "threat_level": "ELEVATED",
        "threat_type": "WARNING",
        "cultural": [
            "Eliminate dusty road edges; dust accumulation suppresses natural predaceous mite populations.",
            "Overhead misting during hottest midday hours raises microclimate humidity, impeding mite reproduction rates.",
            "Identify and rogue heavily webbed 'hot-spot' plants; place directly into plastic bags before field transit."
        ],
        "chemical": [
            "Selective Miticides (Acaricides): Bifenazate (Acramite 50WS, IRAC 20D) @ 500 g/ha OR Spiromesifen (Oberon 2SC, IRAC 23) @ 400 mL/ha.",
            "Translaminar Option: Abamectin (Agri-Mek 0.15EC, IRAC 6) @ 300 mL/ha tank-mixed with paraffinic horticultural oil (0.25% v/v).",
            "IRAC Resistance Protocol: Never apply more than one consecutive treatment of any single IRAC group; rotate between IRAC 6, 10B, 20D, and 23."
        ],
        "preventative": [
            "Biological Vector Control: Inoculate canopy with predatory phytoseiid mites (Phytoseiulus persimilis @ 5 mites/m² or Neoseiulus californicus @ 10 mites/m²).",
            "Avoid broad-spectrum synthetic pyrethroids or carbamates that eliminate predatory beneficial arthropods."
        ]
    },
    11: {
        "condition": "Tomato: Target Spot (Corynespora cassiicola)",
        "pathogen": "Corynespora cassiicola (Pleosporales)",
        "threat_level": "MODERATE",
        "threat_type": "WARNING",
        "cultural": [
            "Elevate lower canopy via sucker removal; maximize air movement through lower rows.",
            "Avoid excessive ammoniacal nitrogen fertilizer applications that generate soft, succulent vegetative canopies prone to rapid fungal ingress.",
            "Sanitize all field harvesting crates and equipment with 0.5% peracetic acid solution."
        ],
        "chemical": [
            "Targeted Fungicide Therapy: Fluxapyroxad + Pyraclostrobin (Priaxor, FRAC 7 + 11) @ 350 mL/ha.",
            "Alternative FRAC Rotation: Boscalid (Endura, FRAC 7) @ 500 g/ha alternating with Chlorothalonil (FRAC M05) @ 2.0 L/ha.",
            "Application Interval: 7 to 10 days depending on local canopy wetness duration."
        ],
        "preventative": [
            "Enforce strict destruction of alternate weed hosts including Commelina diffusa and Euphorbia heterophylla.",
            "Implement a minimum 2-year crop rotation with non-host monocots (Sweet corn, Sorghum)."
        ]
    },
    12: {
        "condition": "Tomato: Tomato Yellow Leaf Curl Virus (TYLCV)",
        "pathogen": "Begomovirus TYLCV (Vector: Bemisia tabaci)",
        "threat_level": "CRITICAL EMERGENCY",
        "threat_type": "CRITICAL",
        "cultural": [
            "No chemical cure exists for viral particles in planta; management is strictly targeted at vector (Bemisia tabaci) exclusion.",
            "Eradicate and bag all symptomatic stunted plants showing upward leaf curling and interveinal chlorosis immediately to minimize field inoculum.",
            "Install 50-mesh insect-exclusion screens on all greenhouse intake and exhaust louvers."
        ],
        "chemical": [
            "Systemic Vector Suppression (Soil Drench at Transplant): Imidacloprid (Admire Pro, IRAC 4A) @ 500 mL/ha OR Thiamethoxam (Platinum 75SG, IRAC 4A).",
            "Foliar Vector Knockdown: Flupyradifurone (Sivanto Prime, IRAC 4D) @ 750 mL/ha alternating with Cyantraniliprole (Verimark, IRAC 28) @ 600 mL/ha.",
            "Insect Growth Regulator: Pyriproxyfen (Knack, IRAC 7C) @ 600 mL/ha to eliminate whitefly nymphal instars."
        ],
        "preventative": [
            "Plant exclusively certified TYLCV-resistant hybrids possessing Ty-1, Ty-2, or Ty-3 resistant gene complexes (e.g., 'Tygress', 'Charger').",
            "Enforce a mandatory 2-month region-wide host-free period between commercial tomato seasons."
        ]
    },
    13: {
        "condition": "Tomato: Tomato Mosaic Virus (ToMV)",
        "pathogen": "Tobamovirus ToMV (Mechanical Transmission)",
        "threat_level": "HIGH THREAT",
        "threat_type": "WARNING",
        "cultural": [
            "ToMV is extremely stable and easily spread via hands, tools, clothing, and irrigation lines; no chemical cure exists.",
            "Workers must wash hands and dip tools in 20% non-fat dry milk solution or 10% trisodium phosphate (TSP) every 30 minutes during pruning.",
            "Strictly ban all tobacco product use by workers within 50 meters of greenhouse production areas.",
            "Carefully remove infected plants along with surrounding root balls in sealed containers."
        ],
        "chemical": [
            "Chemical virucides are ineffective against internal systemic viral titers.",
            "Surface Disinfection: Disinfect all metal trellis wires and greenhouse frames with 0.5% sodium hypochlorite solution or Virkon-S (1% w/v)."
        ],
        "preventative": [
            "Select modern commercial hybrids possessing homozygous Tm-2 or Tm-2^2 resistance loci.",
            "Steam-sterilize or solarize greenhouse soils at >= 70°C for 60 minutes prior to transplanting."
        ]
    },
    14: {
        "condition": "Tomato: Physiological Equilibrium (Healthy Foliage)",
        "pathogen": "Physiologically Normal (Zero Microbial Lesions)",
        "threat_level": "NOMINAL",
        "threat_type": "NOMINAL",
        "cultural": [
            "Maintain balanced electrical conductivity (EC 2.2 - 2.8 mS/cm) and rootzone pH (5.8 - 6.2).",
            "Perform weekly leaf de-leafing up to lowest ripening fruit cluster to maintain aerodynamic canopy microclimates."
        ],
        "chemical": [
            "Zero curative chemical intervention required.",
            "Preventative Biological Defense: Inoculate rootzone with Trichoderma harzianum strain T-22 @ 1.0 kg/ha to prime systemic induced resistance against soilborne pathogens."
        ],
        "preventative": [
            "Maintain computerized environmental logs for vapor pressure deficit (VPD target: 0.8 - 1.2 kPa).",
            "Execute systematic weekly scouting transects across all commercial zones."
        ]
    }
}

def lookup_clinical_protocol(top_diag):
    """
    Deterministic protocol synthesis.
    Maps diagnosis to agronomic intervention cards.
    """
    cls_idx = top_diag.get("class_idx", 0) if top_diag else 0
    proto = CLINICAL_TREATMENT_PROTOCOLS.get(cls_idx, CLINICAL_TREATMENT_PROTOCOLS[0])
    return {
        "condition": proto["condition"],
        "pathogen": proto["pathogen"],
        "threat_level": proto["threat_level"],
        "threat_type": proto["threat_type"],
        "sections": {
            "section_01": {
                "title": "1. Cultural Management & Sanitation",
                "items": proto.get("cultural", []),
            },
            "section_02": {
                "title": "2. Biological & Chemical Interventions (FRAC / IRAC)",
                "items": proto.get("chemical", []),
            },
            "section_03": {
                "title": "3. Vector Exclusion & Crop Rotation",
                "items": proto.get("preventative", proto.get("vector", [])),
            },
        }
    }

# ==============================================================================
# 5. CONVNEXT-TINY NEURAL ARCHITECTURES & RESIDUAL REFINEMENT
# ==============================================================================
class SafeToTensor(nn.Module):
    def forward(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic
        pic = pic.convert("RGB")
        w, h = pic.size
        bt = torch.frombuffer(bytearray(pic.tobytes()), dtype=torch.uint8)
        return bt.view(h, w, 3).permute(2, 0, 1).float().div(255.0)

TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BILINEAR),
    SafeToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

# ==============================================================================
# 6. NEURAL NETWORK ARCHITECTURAL DEFINITIONS
# ==============================================================================
class ConvNeXtTiny_ConvBNReLU(nn.Module):
    def __init__(self, num_classes=15, dropout_p=0.0):
        super().__init__()
        self.features = models.convnext_tiny(weights=None).features
        self.refinement = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(768, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.refinement(feat) + feat
        return self.classifier(self.pool(feat))

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
        return x * self.fc(y).view(b, c, 1, 1)

class ConvNeXtTiny_SE(nn.Module):
    def __init__(self, num_classes=15, dropout_p=0.0, reduction=16):
        super().__init__()
        self.features = models.convnext_tiny(weights=None).features
        self.pre_conv = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.refinement = SEBlock(channels=768, reduction=reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(768, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        refined = self.pre_conv(feat)
        excited = self.refinement(refined) + feat
        return self.classifier(self.pool(excited))

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
        avg_pool = x.mean(dim=(2, 3))
        max_pool = x.amax(dim=(2, 3))
        scale = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(b, c, 1, 1)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale

class CBAMBlock(nn.Module):
    def __init__(self, channels=768, reduction_ratio=16):
        super().__init__()
        self.channel_gate = ChannelGate(channels, reduction_ratio)
        self.spatial_gate = SpatialGate()

    def forward(self, x):
        return self.spatial_gate(self.channel_gate(x))

class ConvNeXtTiny_CBAM(nn.Module):
    def __init__(self, num_classes=15, dropout_p=0.0, reduction=16):
        super().__init__()
        self.features = models.convnext_tiny(weights=None).features
        self.pre_conv = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.refinement = CBAMBlock(channels=768, reduction_ratio=reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(768, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        refined = self.pre_conv(feat)
        attended = self.refinement(refined) + feat
        return self.classifier(self.pool(attended))

MODEL_REGISTRY = {
    "se_block": {
        "cls": ConvNeXtTiny_SE,
        "files": ["weights_se_block_fp16.pth", "plant_disease_model_fp16.pth", "weights_se_block.pth"],
    },
    "conv_bn_relu": {
        "cls": ConvNeXtTiny_ConvBNReLU,
        "files": ["weights_conv_bn_relu_fp16.pth", "weights_conv_bn_relu.pth"],
    },
    "cbam": {
        "cls": ConvNeXtTiny_CBAM,
        "files": ["weights_cbam_fp16.pth", "weights_cbam.pth"],
    },
}

# ==============================================================================
# 7. INFERENCE ENGINE (HARDWARE ACCELERATION ENABLED)
# ==============================================================================
class DiagnosticEngine:
    def __init__(self):
        self.device, self.device_name = detect_compute_device()
        self._cache = {}

    def _resolve_weight_path(self, filename):
        base = Path(__file__).resolve().parent
        search_dirs = [
            base / "models",
            base,
            base / "Part_2_Model_Testing_and_GUI" / "models",
            base.parent / "Part_2_Model_Testing_and_GUI" / "models",
            base.parent / "final results" / "models",
            base.parent / "models",
            base.parent / "compatible-model" / "models",
            base.parent / "Compatable_run" / "models",
            base.parent / "Compatible_run" / "models",
        ]
        for d in search_dirs:
            p = d / filename
            if p.exists():
                return p.resolve()
        return None

    def _load_model(self, key):
        if key in self._cache:
            return self._cache[key]

        reg = MODEL_REGISTRY[key]
        resolved = None
        for fn in reg["files"]:
            resolved = self._resolve_weight_path(fn)
            if resolved:
                break

        if not resolved:
            raise FileNotFoundError(f"Missing model weights for {key}: Searched {reg['files']}")

        net = reg["cls"](num_classes=15, dropout_p=0.0)
        sd = torch.load(resolved, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
        net.load_state_dict(sd)
        net.to(self.device).eval()
        self._cache[key] = net
        return net

    @torch.no_grad()
    def analyze(self, img_path, key="se_block", top_k=5):
        net = self._load_model(key)
        img = Image.open(img_path).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(self.device)

        logits = net(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu()
        top_probs, top_indices = torch.topk(probs, k=min(top_k, 15))

        results = []
        for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
            meta = CLASS_TAXONOMY[idx]
            results.append({
                "class_index": idx,
                "plant": meta["plant"],
                "health_status": meta["status"],
                "diagnosis": meta["diag"],
                "pathogen": meta["pathogen"],
                "is_diseased": meta["diseased"],
                "confidence_pct": round(p * 100.0, 2),
                "raw_label": meta["raw"],
            })
        return results

ENGINE = DiagnosticEngine()

# ==============================================================================
# 8. TACTILE ENTERPRISE BUTTON SUBCLASS
# ==============================================================================

# ==============================================================================
# 9. TACTILE SCIENTIFIC UI COMPONENTS
# ==============================================================================
class EnterpriseButton(ctk.CTkButton):
    """
    Custom tactile button with dynamic palette hover response
    and native pointer cursor integration across platforms.
    """
    def __init__(self, master, **kwargs):
        cursor = OSFontResolver.get_cursor()
        super().__init__(master, cursor=cursor, corner_radius=4, **kwargs)
        self._orig_border = kwargs.get("border_color", "transparent")
        self._orig_fg = kwargs.get("fg_color", None)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_enter(self, event=None):
        if str(self.cget("state")) != "disabled":
            self.configure(border_width=1, border_color=THEME["border_focus"])

    def _on_leave(self, event=None):
        if str(self.cget("state")) != "disabled":
            self.configure(border_width=0, border_color=self._orig_border)


class PrecisionRadialGauge(ctk.CTkCanvas):
    """
    Custom animated circular gauge with structural tick marks
    and tabular percentage readouts rendered using active palette colors.
    """
    def __init__(self, master, value=0.0, size=120, thickness=10,
                 color=None, title="ACCURACY", subtitle="72.4%", bg_parent=None, **kw):
        bg = bg_parent or THEME["bg_card"]
        super().__init__(master, width=size, height=size + 30,
                         highlightthickness=0, bg=bg, **kw)
        self.val = 0.0
        self.target = float(value)
        self.sz = size
        self.th = thickness
        self.color = color or THEME["accent"]
        self.title_str = title
        self.subtitle_str = subtitle
        self.bg_parent = bg
        self._render()

    def update_gauge(self, value, color=None, title=None, subtitle=None):
        try:
            if not self.winfo_exists():
                return
        except Exception:
            return
        self.target = float(value)
        if color: self.color = color
        if title: self.title_str = title
        if subtitle: self.subtitle_str = subtitle
        self.val = 0.0
        self._step_animation()

    def _step_animation(self):
        try:
            if not self.winfo_exists():
                return
            if self.val < self.target:
                self.val = min(self.val + max(0.8, (self.target - self.val) * 0.16), self.target)
                self._render()
                self.after(16, self._step_animation)
            else:
                self.val = self.target
                self._render()
        except Exception:
            pass

    def _render(self):
        try:
            if not self.winfo_exists():
                return
            self.delete("all")
            s, t = self.sz, self.th
            cx, cy = s // 2, s // 2
            margin = t // 2 + 6
            radius = (s - margin * 2) / 2

            # Draw structural perimeter tick marks
            for i in range(12):
                angle = i * (math.pi / 6)
                x1 = cx + (radius + t / 2 + 1) * math.cos(angle)
                y1 = cy + (radius + t / 2 + 1) * math.sin(angle)
                x2 = cx + (radius + t / 2 + 4) * math.cos(angle)
                y2 = cy + (radius + t / 2 + 4) * math.sin(angle)
                self.create_line(x1, y1, x2, y2, fill=THEME["border_hairline"], width=1)

            # Background track ring
            self.create_arc(margin, margin, s - margin, s - margin,
                            start=90, extent=-360, style="arc",
                            outline=THEME["bar_track"], width=t)

            # Dynamic arc
            extent = -3.6 * self.val
            if abs(extent) > 0.5:
                self.create_arc(margin, margin, s - margin, s - margin,
                                start=90, extent=extent, style="arc",
                                outline=self.color, width=t)

            # Monospace tabular percentage readout
            self.create_text(cx, cy - 1, text=f"{self.val:.1f}%",
                             fill=THEME["text_primary"], font=OSFontResolver.get_mono(14, "bold"))

            # Metadata labels
            self.create_text(cx, s + 6, text=self.title_str,
                             fill=THEME["text_secondary"], font=OSFontResolver.get_ui(8, "bold"))
            self.create_text(cx, s + 19, text=self.subtitle_str,
                             fill=THEME["text_muted"], font=OSFontResolver.get_mono(8, "normal"))
        except Exception:
            pass


class PrecisionProgressBar(ctk.CTkCanvas):
    """
    Linear horizontal distribution bar with monospace readouts
    synchronized with active palette surfaces.
    """
    def __init__(self, master, label="", value=0.0, color=None, height=26, bg_parent=None, **kw):
        bg = bg_parent or THEME["bg_card"]
        super().__init__(master, height=height, highlightthickness=0, bg=bg, **kw)
        self.label_str = label
        self.target_pct = float(value)
        self.current_pct = 0.0
        self.fill_color = color or THEME["accent"]
        self.bar_h = height
        self.bind("<Configure>", self._draw)

    def set_progress(self, val, label=None):
        try:
            if not self.winfo_exists():
                return
        except Exception:
            return
        if label: self.label_str = label
        self.target_pct = float(val)
        self.current_pct = 0.0
        self._animate()

    def _animate(self):
        try:
            if not self.winfo_exists():
                return
            if self.current_pct < self.target_pct:
                self.current_pct = min(self.current_pct + max(0.8, (self.target_pct - self.current_pct) * 0.16), self.target_pct)
                self._draw()
                self.after(16, self._animate)
            else:
                self.current_pct = self.target_pct
                self._draw()
        except Exception:
            pass

    def _draw(self, event=None):
        try:
            if not self.winfo_exists():
                return
            self.delete("all")
            w = self.winfo_width()
            h = self.bar_h
            if w < 10: return

            pad = 2
            # Track background with subtle hairline outline
            self.create_rectangle(0, pad, w, h - pad, fill=THEME["bar_track"], outline=THEME["border_hairline"], width=1)

            # Value bar
            fill_w = max(0, int(w * (self.current_pct / 100.0)))
            if fill_w > 0:
                self.create_rectangle(1, pad + 1, fill_w, h - pad - 1, fill=self.fill_color, outline="", width=0)

            # Label and tabular percentage
            self.create_text(8, h // 2, text=self.label_str, anchor="w",
                             fill=THEME["text_primary"], font=OSFontResolver.get_ui(10, "bold"))
            self.create_text(w - 8, h // 2, text=f"{self.current_pct:.1f}%", anchor="e",
                             fill=THEME["text_primary"], font=OSFontResolver.get_mono(10, "bold"))
        except Exception:
            pass


# ==============================================================================
# 10. MAIN BOTANICAL WORKSTATION APPLICATION
# ==============================================================================
class PhytoScanApp(ctk.CTk):
    """
    Deterministic Botanical Pathology & Agronomic Diagnostic Workstation.
    Incorporates 8 Sanzo Wada traditional Japanese color combinations
    with native and in-window menu bar palette switching.
    """
    def __init__(self):
        ctk.set_appearance_mode("Dark")
        super().__init__()

        self.title("PhytoScan · Botanical Pathology Diagnostic Workstation")
        self.geometry("1220x820")
        self.minsize(1040, 680)

        # Application State
        self.active_image_path = None
        self._image_refs = []
        self._is_busy = False
        self._latest_report_data = None
        self.current_palette_key = "wada_182"
        self.model_radios = []

        # Build interface and native menu
        self._build_menu_bar()
        self._build_interface()
        self._bind_cross_platform_events()

        # Check for bundled sample image
        sample_path = self._find_sample_leaf()
        if sample_path:
            self._load_specimen_file(sample_path)

    # --------------------------------------------------------------------------
    # Native Application Menu Bar
    # --------------------------------------------------------------------------
    def _build_menu_bar(self):
        self.menubar = tk.Menu(self)

        # 1. File Menu
        file_menu = tk.Menu(self.menubar, tearoff=0)
        file_menu.add_command(label="Open Image... [Ctrl+O]", command=self._on_browse_specimen)
        file_menu.add_command(
            label="Load Sample Leaf",
            command=lambda: self._load_specimen_file(self._find_sample_leaf()) if self._find_sample_leaf() else None
        )
        file_menu.add_command(label="Save Diagnostic Report [Ctrl+S]", command=self._on_export_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        self.menubar.add_cascade(label="File", menu=file_menu)

        # 2. Color Palette Menu (Sanzo Wada Japanese Color Theory)
        self.palette_menu_var = tk.StringVar(value="wada_182")
        palette_menu = tk.Menu(self.menubar, tearoff=0)
        for key, pdata in SANZO_WADA_PALETTES.items():
            palette_menu.add_radiobutton(
                label=pdata["menu_label"],
                variable=self.palette_menu_var,
                value=key,
                command=lambda k=key: self.apply_palette(k)
            )
        self.menubar.add_cascade(label="Color Palette", menu=palette_menu)

        # 3. Help Menu
        help_menu = tk.Menu(self.menubar, tearoff=0)
        help_menu.add_command(label="About PhytoScan", command=self._show_about_dialog)
        self.menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=self.menubar)

    def _show_about_dialog(self):
        msg = (
            "PhytoScan · Botanical Pathology Diagnostic Workstation\n"
            "Version: 4.2 (Authentic Desktop Edition)\n\n"
            "• Neural Architecture: ConvNeXt-Tiny Deep Residual Feature Backbone\n"
            "• Solanaceous Taxonomy: 15 Classes across Pepper Bell, Potato, Tomato\n"
            "• Color Harmonies: Referencing 'A Dictionary of Color Combinations' by Sanzo Wada (1930s)\n"
            "• Local Deterministic Engine: Zero external API requirements or network calls."
        )
        messagebox.showinfo("About PhytoScan", msg)

    # --------------------------------------------------------------------------
    # Master Interface Layout
    # --------------------------------------------------------------------------
    def _build_interface(self):
        self.configure(fg_color=THEME["bg_root"])

        # 1. Top Application Header
        self.header_frame = ctk.CTkFrame(self, height=52, fg_color=THEME["bg_panel"], corner_radius=0)
        self.header_frame.pack(fill="x")
        self.header_frame.pack_propagate(False)

        self.brand_lbl = ctk.CTkLabel(
            self.header_frame, text="PhytoScan",
            font=OSFontResolver.get_ui(16, "bold"),
            text_color=THEME["text_primary"]
        )
        self.brand_lbl.pack(side="left", padx=(18, 10))

        self.sub_lbl = ctk.CTkLabel(
            self.header_frame, text="|   Plant Pathology Diagnostics",
            font=OSFontResolver.get_ui(11, "normal"),
            text_color=THEME["text_muted"]
        )
        self.sub_lbl.pack(side="left")

        # Right side header: Version tag & In-window Palette Dropdown
        self.version_tag = ctk.CTkLabel(
            self.header_frame, text=f"v4.2  •  {ENGINE.device_name}",
            font=OSFontResolver.get_mono(9, "normal"),
            text_color=THEME["text_muted"]
        )
        self.version_tag.pack(side="right", padx=(10, 18))

        palette_options = [p["menu_label"] for p in SANZO_WADA_PALETTES.values()]
        self.palette_dropdown = ctk.CTkOptionMenu(
            self.header_frame,
            values=palette_options,
            command=self._on_palette_dropdown_selected,
            font=OSFontResolver.get_ui(10, "normal"),
            dropdown_font=OSFontResolver.get_ui(10, "normal"),
            width=290, height=28,
            fg_color=THEME["bg_card"],
            button_color=THEME["accent"],
            button_hover_color=THEME["accent_hover"],
            text_color=THEME["text_primary"],
            dropdown_fg_color=THEME["bg_panel"],
            dropdown_hover_color=THEME["accent_dim"],
            dropdown_text_color=THEME["text_primary"],
            corner_radius=4
        )
        self.palette_dropdown.set(SANZO_WADA_PALETTES["wada_182"]["menu_label"])
        self.palette_dropdown.pack(side="right", padx=(6, 12))

        self.palette_lbl = ctk.CTkLabel(
            self.header_frame, text="Palette:",
            font=OSFontResolver.get_ui(11, "normal"),
            text_color=THEME["text_secondary"]
        )
        self.palette_lbl.pack(side="right")

        # 2. Main Two-Column Layout
        self.main_body = ctk.CTkFrame(self, fg_color=THEME["bg_root"], corner_radius=0)
        self.main_body.pack(fill="both", expand=True, padx=10, pady=8)
        self.main_body.grid_columnconfigure(0, weight=1, minsize=360)
        self.main_body.grid_columnconfigure(1, weight=2, minsize=680)
        self.main_body.grid_rowconfigure(0, weight=1)

        # ----------------------------------------------------------------------
        # LEFT CONTROL PANEL (Specimen, Model Selection, Execution)
        # ----------------------------------------------------------------------
        self.left_panel = ctk.CTkFrame(
            self.main_body, fg_color=THEME["bg_panel"], corner_radius=4,
            border_width=1, border_color=THEME["border_hairline"]
        )
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # Section 1: Specimen Acquisition
        self.sec1_hdr = ctk.CTkLabel(
            self.left_panel, text="SPECIMEN IMAGE",
            font=OSFontResolver.get_mono(10, "bold"),
            text_color=THEME["text_muted"]
        )
        self.sec1_hdr.pack(anchor="w", padx=16, pady=(14, 4))

        # Specimen Image Container
        self.specimen_frame = ctk.CTkFrame(
            self.left_panel, height=210, fg_color=THEME["bg_card"],
            corner_radius=4, border_width=1, border_color=THEME["border_hairline"]
        )
        self.specimen_frame.pack(fill="x", padx=14, pady=4)
        self.specimen_frame.pack_propagate(False)

        self.specimen_label = ctk.CTkLabel(self.specimen_frame, text="", fg_color="transparent")
        self.specimen_label.pack(expand=True)

        self.placeholder_text = ctk.CTkLabel(
            self.specimen_frame,
            text="No specimen loaded\nClick 'Choose Photo...' or press [Ctrl+O]\nto select a leaf image for analysis",
            font=OSFontResolver.get_ui(11, "normal"),
            text_color=THEME["text_muted"],
            justify="center"
        )
        self.placeholder_text.place(relx=0.5, rely=0.5, anchor="center")

        # Active File Tag
        self.filename_tag = ctk.CTkLabel(
            self.left_panel, text="No file selected",
            font=OSFontResolver.get_mono(9, "normal"),
            text_color=THEME["text_secondary"], wraplength=340
        )
        self.filename_tag.pack(pady=(2, 6))

        # Action Buttons Row
        btn_row = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        btn_row.pack(fill="x", padx=14, pady=2)

        self.load_btn = EnterpriseButton(
            btn_row, text="Choose Photo...  [Ctrl+O]",
            font=OSFontResolver.get_ui(11, "bold"),
            height=36, fg_color=THEME["accent"],
            hover_color=THEME["accent_hover"],
            command=self._on_browse_specimen
        )
        self.load_btn.pack(side="left", expand=True, fill="x", padx=(0, 3))

        sample_p = self._find_sample_leaf()
        if sample_p:
            self.sample_btn = EnterpriseButton(
                btn_row, text="Sample Leaf",
                font=OSFontResolver.get_ui(11, "normal"),
                height=36, fg_color=THEME["bg_card_sub"],
                hover_color=THEME["border_hairline"],
                command=lambda: self._load_specimen_file(sample_p)
            )
            self.sample_btn.pack(side="right", expand=True, fill="x", padx=(3, 0))

        # Section 2: Model Architecture Selection
        self.sec2_hdr = ctk.CTkLabel(
            self.left_panel, text="MODEL SELECTION",
            font=OSFontResolver.get_mono(10, "bold"),
            text_color=THEME["text_muted"]
        )
        self.sec2_hdr.pack(anchor="w", padx=16, pady=(18, 6))

        self.pipeline_var = ctk.StringVar(value="all")
        models_options = [
            ("All 3 Models (Cross-Verification)", "all"),
            ("ConvNeXt-SE (Primary · 72.4% Acc)", "se_block"),
            ("ConvNeXt-Residual (3x3 Conv · 68.9% Acc)", "conv_bn_relu"),
            ("ConvNeXt-CBAM (Dual Attention · 67.4% Acc)", "cbam"),
        ]
        self.model_radios = []
        for opt_lbl, opt_val in models_options:
            rb = ctk.CTkRadioButton(
                self.left_panel, text=opt_lbl, variable=self.pipeline_var, value=opt_val,
                font=OSFontResolver.get_ui(11, "normal"),
                text_color=THEME["text_secondary"],
                fg_color=THEME["accent"],
                hover_color=THEME["accent_hover"],
                border_color=THEME["border_hairline"]
            )
            rb.pack(anchor="w", padx=20, pady=3)
            self.model_radios.append(rb)

        # Primary Execution Button
        self.exec_btn = EnterpriseButton(
            self.left_panel, text="Analyze Leaf Photo  [Enter]",
            font=OSFontResolver.get_ui(12, "bold"),
            height=44, fg_color=THEME["accent"],
            hover_color=THEME["accent_hover"],
            command=self._execute_diagnosis
        )
        self.exec_btn.pack(fill="x", padx=14, pady=(24, 6))

        # Real-time state label
        self.status_msg = ctk.CTkLabel(
            self.left_panel, text="Ready — Select a leaf photo to begin",
            font=OSFontResolver.get_mono(9, "normal"),
            text_color=THEME["text_muted"]
        )
        self.status_msg.pack(pady=(0, 10))

        # ----------------------------------------------------------------------
        # RIGHT REPORT VIEWPORT
        # ----------------------------------------------------------------------
        self.right_panel = ctk.CTkFrame(self.main_body, fg_color=THEME["bg_root"], corner_radius=0)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        self.scroll_viewport = ctk.CTkScrollableFrame(
            self.right_panel, fg_color=THEME["bg_root"], corner_radius=0,
            scrollbar_button_color=THEME["border_hairline"],
            scrollbar_button_hover_color=THEME["accent"]
        )
        self.scroll_viewport.pack(fill="both", expand=True)

        self._render_standby_view()

    # --------------------------------------------------------------------------
    # Palette Switching Engine (Sanzo Wada Harmonies)
    # --------------------------------------------------------------------------
    def _on_palette_dropdown_selected(self, choice_label):
        for k, p in SANZO_WADA_PALETTES.items():
            if p["menu_label"] == choice_label:
                self.apply_palette(k)
                break

    def apply_palette(self, palette_key):
        """
        Dynamically applies a Sanzo Wada color combination palette
        across the entire workstation interface.
        """
        if palette_key not in SANZO_WADA_PALETTES:
            return

        self.current_palette_key = palette_key
        pal = SANZO_WADA_PALETTES[palette_key]

        THEME.clear()
        THEME.update(pal)

        # Update model benchmark color assignments
        BENCHMARK_PROFILES["se_block"]["color"] = THEME["accent"]
        BENCHMARK_PROFILES["conv_bn_relu"]["color"] = THEME["secondary"]
        BENCHMARK_PROFILES["cbam"]["color"] = THEME["tertiary"]

        # Synchronize menu selectors
        self.palette_menu_var.set(palette_key)
        self.palette_dropdown.set(pal["menu_label"])

        # Reconfigure root & master frames
        self.configure(fg_color=THEME["bg_root"])
        self.header_frame.configure(fg_color=THEME["bg_panel"])
        self.brand_lbl.configure(text_color=THEME["text_primary"])
        self.sub_lbl.configure(text_color=THEME["text_muted"])
        self.palette_lbl.configure(text_color=THEME["text_secondary"])
        self.palette_dropdown.configure(
            fg_color=THEME["bg_card"],
            button_color=THEME["accent"],
            button_hover_color=THEME["accent_hover"],
            text_color=THEME["text_primary"],
            dropdown_fg_color=THEME["bg_panel"],
            dropdown_hover_color=THEME["accent_dim"],
            dropdown_text_color=THEME["text_primary"]
        )
        self.version_tag.configure(text_color=THEME["text_muted"])

        self.main_body.configure(fg_color=THEME["bg_root"])
        self.left_panel.configure(fg_color=THEME["bg_panel"], border_color=THEME["border_hairline"])
        self.sec1_hdr.configure(text_color=THEME["text_muted"])
        self.sec2_hdr.configure(text_color=THEME["text_muted"])
        self.specimen_frame.configure(fg_color=THEME["bg_card"], border_color=THEME["border_hairline"])
        self.placeholder_text.configure(text_color=THEME["text_muted"])
        self.filename_tag.configure(text_color=THEME["text_secondary"])

        self.load_btn.configure(
            fg_color=THEME["accent"],
            hover_color=THEME["accent_hover"],
            text_color=THEME["text_primary"]
        )
        if hasattr(self, "sample_btn"):
            self.sample_btn.configure(
                fg_color=THEME["bg_card_sub"],
                hover_color=THEME["border_hairline"],
                text_color=THEME["text_primary"]
            )

        for rb in self.model_radios:
            rb.configure(
                text_color=THEME["text_secondary"],
                fg_color=THEME["accent"],
                hover_color=THEME["accent_hover"],
                border_color=THEME["border_hairline"]
            )

        self.exec_btn.configure(
            fg_color=THEME["accent"],
            hover_color=THEME["accent_hover"],
            text_color=THEME["text_primary"]
        )
        self.status_msg.configure(text_color=THEME["text_muted"])

        self.right_panel.configure(fg_color=THEME["bg_root"])
        self.scroll_viewport.configure(
            fg_color=THEME["bg_root"],
            scrollbar_button_color=THEME["border_hairline"],
            scrollbar_button_hover_color=THEME["accent"]
        )

        # Refresh viewport contents
        if self._latest_report_data:
            self._render_diagnostic_results(
                self._latest_report_data["results"],
                self._latest_report_data["protocol"]
            )
        else:
            self._render_standby_view()

    # --------------------------------------------------------------------------
    # Standby View (Before diagnosis execution)
    # --------------------------------------------------------------------------
    def _render_standby_view(self):
        for w in self.scroll_viewport.winfo_children():
            w.destroy()

        standby_box = ctk.CTkFrame(
            self.scroll_viewport, fg_color=THEME["bg_panel"],
            corner_radius=4, border_width=1, border_color=THEME["border_hairline"]
        )
        standby_box.pack(fill="x", padx=4, pady=4)

        ctk.CTkLabel(
            standby_box, text="Plant Pathology Diagnostics",
            font=OSFontResolver.get_ui(16, "bold"),
            text_color=THEME["text_primary"]
        ).pack(pady=(24, 4))

        ctk.CTkLabel(
            standby_box,
            text="Load a leaf photograph to analyze diseases across bell pepper, potato, and tomato crops.\n"
                 "Deep convolutional neural networks are initialized locally in memory.",
            font=OSFontResolver.get_ui(11, "normal"),
            text_color=THEME["text_secondary"], justify="center"
        ).pack(pady=(0, 20))

        # Model Performance Overview Cards
        models_row = ctk.CTkFrame(standby_box, fg_color="transparent")
        models_row.pack(fill="x", padx=16, pady=(0, 20))

        for key in ["se_block", "conv_bn_relu", "cbam"]:
            prof = BENCHMARK_PROFILES[key]
            card = ctk.CTkFrame(
                models_row, fg_color=THEME["bg_card"], corner_radius=4,
                border_width=1, border_color=THEME["border_hairline"]
            )
            card.pack(side="left", expand=True, fill="both", padx=4, pady=4)

            ctk.CTkLabel(
                card, text=prof["short_name"],
                font=OSFontResolver.get_ui(11, "bold"),
                text_color=prof["color"]
            ).pack(pady=(12, 2))

            ctk.CTkLabel(
                card, text=f"Validation Accuracy: {prof['acc']:.2f}%",
                font=OSFontResolver.get_ui(13, "bold"),
                text_color=THEME["text_primary"]
            ).pack()

            ctk.CTkLabel(
                card, text=f"Sensitivity: {prof['sensitivity']:.2f}%  |  Parameters: {prof['params']}",
                font=OSFontResolver.get_mono(9, "normal"),
                text_color=THEME["text_muted"]
            ).pack(pady=(2, 12))

        # Clean Information Footer
        info_f = ctk.CTkFrame(standby_box, fg_color=THEME["bg_card_sub"], corner_radius=0)
        info_f.pack(fill="x", padx=16, pady=(0, 16))

        ctk.CTkLabel(
            info_f,
            text="Supported Crops: Bell Pepper, Potato, Tomato  •  15 Pathological Classes Recognized\n"
                 "Deterministic Offline Engine  •  Zero External Network Transmission",
            font=OSFontResolver.get_mono(9, "normal"),
            text_color=THEME["text_muted"], justify="center"
        ).pack(pady=8)

    # --------------------------------------------------------------------------
    # Event Handlers & Specimen Acquisition
    # --------------------------------------------------------------------------
    def _find_sample_leaf(self):
        base = Path(__file__).resolve().parent
        for cand in [
            base / "sample_leaf.jpg",
            base.parent / "sample_leaf.jpg",
            base / "web_app" / "sample_leaf.jpg",
            base.parent / "final results" / "web_app" / "sample_leaf.jpg",
        ]:
            if cand.exists():
                return str(cand.resolve())
        return None

    def _on_browse_specimen(self, event=None):
        if self._is_busy:
            return
        path = filedialog.askopenfilename(
            title="Select Plant Leaf Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All Files", "*.*")]
        )
        if path:
            self._load_specimen_file(path)

    def _load_specimen_file(self, path):
        if self._is_busy:
            return
        self.active_image_path = path
        self.filename_tag.configure(text=f"File: {Path(path).name}")
        self.placeholder_text.place_forget()

        img = Image.open(path).convert("RGB")
        img.thumbnail((340, 200), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        self._image_refs.append(tk_img)
        self.specimen_label.configure(image=tk_img)
        self.status_msg.configure(text=f"Loaded: {Path(path).name}", text_color=THEME["accent"])

    def _bind_cross_platform_events(self):
        # Keyboard shortcuts
        self.bind_all("<Control-o>", self._on_browse_specimen)
        self.bind_all("<Command-o>", self._on_browse_specimen)
        self.bind_all("<Control-s>", lambda e: self._on_export_report())
        self.bind_all("<Command-s>", lambda e: self._on_export_report())
        self.bind_all("<Return>", lambda e: self._execute_diagnosis())
        self.bind_all("<Control-Return>", lambda e: self._execute_diagnosis())

        # Cross-platform mousewheel handling
        def _on_mousewheel(event):
            try:
                canvas = self.scroll_viewport._parent_canvas
                if sys.platform == "darwin":
                    delta = -1 * event.delta
                elif sys.platform == "win32":
                    delta = -1 * (event.delta // 120)
                else:
                    delta = -1 * (event.delta // 120) if event.delta else 0
                canvas.yview_scroll(delta, "units")
            except Exception:
                pass

        def _on_linux_up(event):
            try:
                self.scroll_viewport._parent_canvas.yview_scroll(-1, "units")
            except Exception:
                pass

        def _on_linux_down(event):
            try:
                self.scroll_viewport._parent_canvas.yview_scroll(1, "units")
            except Exception:
                pass

        self.bind_all("<MouseWheel>", _on_mousewheel)
        self.bind_all("<Button-4>", _on_linux_up)
        self.bind_all("<Button-5>", _on_linux_down)

    # --------------------------------------------------------------------------
    # Diagnostic Execution Workflow
    # --------------------------------------------------------------------------
    def _execute_diagnosis(self):
        if self._is_busy:
            return
        if not self.active_image_path:
            self.status_msg.configure(text="Please select an image first", text_color=THEME["status_warning_txt"])
            return

        self._is_busy = True
        self.exec_btn.configure(state="disabled", text="Analyzing Image...")
        self.load_btn.configure(state="disabled")
        if hasattr(self, "sample_btn"):
            self.sample_btn.configure(state="disabled")
        self.status_msg.configure(text="Running neural evaluation...", text_color=THEME["accent"])
        self.update_idletasks()

        selection = self.pipeline_var.get()
        keys = ["se_block", "conv_bn_relu", "cbam"] if selection == "all" else [selection]

        def worker():
            results = {}
            for k in keys:
                try:
                    results[k] = ENGINE.analyze(self.active_image_path, key=k, top_k=5)
                except Exception as e:
                    results[k] = [{"error": str(e)}]

            # Primary model for care protocol derivation
            primary_key = "se_block" if "se_block" in results else keys[0]
            top_diag = results[primary_key][0] if results[primary_key] and "error" not in results[primary_key][0] else {}

            protocol_data = lookup_clinical_protocol(top_diag)

            self.after(0, lambda: self._render_diagnostic_results(results, protocol_data))

        threading.Thread(target=worker, daemon=True).start()

    # --------------------------------------------------------------------------
    # Render Diagnostic Results & Care Guidelines Matrix
    # --------------------------------------------------------------------------
    def _render_diagnostic_results(self, results, protocol_data):
        self._is_busy = False
        self.exec_btn.configure(state="normal", text="Analyze Leaf Photo  [Enter]")
        self.load_btn.configure(state="normal")
        if hasattr(self, "sample_btn"):
            self.sample_btn.configure(state="normal")
        self.status_msg.configure(text="Analysis complete", text_color=THEME["status_nominal_txt"])

        self._latest_report_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "specimen_path": str(self.active_image_path),
            "results": results,
            "protocol": protocol_data,
        }

        for w in self.scroll_viewport.winfo_children():
            w.destroy()

        def _safe_update_gauge(g, v):
            try:
                if g.winfo_exists():
                    g.update_gauge(v)
            except Exception:
                pass

        def _safe_set_progress(b, v):
            try:
                if b.winfo_exists():
                    b.set_progress(v)
            except Exception:
                pass

        # SECTION 1: Diagnostic Assessment Cards
        ctk.CTkLabel(
            self.scroll_viewport, text="DIAGNOSTIC FINDINGS",
            font=OSFontResolver.get_mono(10, "bold"),
            text_color=THEME["text_muted"]
        ).pack(anchor="w", padx=6, pady=(6, 2))

        cards_row = ctk.CTkFrame(self.scroll_viewport, fg_color="transparent")
        cards_row.pack(fill="x", padx=2)

        for key, preds in results.items():
            if not preds or "error" in preds[0]:
                continue
            top = preds[0]
            prof = BENCHMARK_PROFILES[key]

            is_dis = top["is_diseased"]
            status_text = "Disease Detected" if is_dis else "Healthy Foliage"
            status_color = THEME["status_critical_txt"] if is_dis else THEME["status_nominal_txt"]
            status_bg = THEME["status_critical_bg"] if is_dis else THEME["status_nominal_bg"]
            status_border = THEME["status_critical"] if is_dis else THEME["status_nominal"]

            card = ctk.CTkFrame(
                cards_row, fg_color=THEME["bg_card"], corner_radius=4,
                border_width=1, border_color=THEME["border_hairline"]
            )
            card.pack(side="left", expand=True, fill="both", padx=3, pady=3)

            # Model header
            ctk.CTkLabel(
                card, text=prof["short_name"],
                font=OSFontResolver.get_ui(10, "bold"),
                text_color=prof["color"]
            ).pack(pady=(10, 2))

            # Status Badge
            badge_f = ctk.CTkFrame(
                card, fg_color=status_bg, corner_radius=3,
                border_width=1, border_color=status_border
            )
            badge_f.pack(padx=8, pady=3)
            ctk.CTkLabel(
                badge_f, text=status_text,
                font=OSFontResolver.get_ui(9, "bold"),
                text_color=status_color
            ).pack(padx=8, pady=2)

            # Plant & Disease Identity
            ctk.CTkLabel(
                card, text=top["plant"].upper(),
                font=OSFontResolver.get_ui(15, "bold"),
                text_color=THEME["text_primary"]
            ).pack(pady=(4, 0))

            ctk.CTkLabel(
                card, text=top["diagnosis"],
                font=OSFontResolver.get_ui(11, "bold"),
                text_color=THEME["text_secondary"]
            ).pack(pady=(0, 2))

            ctk.CTkLabel(
                card, text=f"Taxon: {top['pathogen']}",
                font=OSFontResolver.get_mono(8, "normal"),
                text_color=THEME["text_muted"], wraplength=260
            ).pack(pady=(0, 6))

            # Radial Confidence Gauge
            gauge = PrecisionRadialGauge(
                card, value=top["confidence_pct"], size=116, thickness=10,
                color=prof["color"], title="CONFIDENCE", subtitle=f"{top['confidence_pct']:.1f}%",
                bg_parent=THEME["bg_card"]
            )
            gauge.pack(pady=2)
            self.after(150, lambda g=gauge, v=top["confidence_pct"]: _safe_update_gauge(g, v))

            # Probability Distribution for Alternative Classes
            ctk.CTkLabel(
                card, text="ALTERNATIVE PREDICTIONS",
                font=OSFontResolver.get_mono(8, "bold"),
                text_color=THEME["text_muted"]
            ).pack(anchor="w", padx=10, pady=(6, 2))

            for p in preds[:4]:
                pbar = PrecisionProgressBar(
                    card, label=f"{p['plant']}: {p['diagnosis']}",
                    value=p["confidence_pct"], color=prof["color"], height=22,
                    bg_parent=THEME["bg_card"]
                )
                pbar.pack(fill="x", padx=8, pady=1)
                self.after(300, lambda b=pbar, v=p["confidence_pct"]: _safe_set_progress(b, v))

            ctk.CTkFrame(card, height=8, fg_color="transparent").pack()

        # SECTION 2: Model Benchmark Comparison Dials
        if len(results) > 1:
            ctk.CTkLabel(
                self.scroll_viewport, text="MODEL BENCHMARK COMPARISON (36,351 TEST IMAGES)",
                font=OSFontResolver.get_mono(10, "bold"),
                text_color=THEME["text_muted"]
            ).pack(anchor="w", padx=6, pady=(14, 2))

            bench_frame = ctk.CTkFrame(
                self.scroll_viewport, fg_color=THEME["bg_card"],
                corner_radius=4, border_width=1, border_color=THEME["border_hairline"]
            )
            bench_frame.pack(fill="x", padx=2, pady=2)

            dials_box = ctk.CTkFrame(bench_frame, fg_color="transparent")
            dials_box.pack(fill="x", padx=8, pady=8)

            metric_configs = [
                ("OVERALL ACCURACY", "acc"),
                ("DISEASE SENSITIVITY", "sensitivity"),
                ("PRECISION SPECIFICITY", "precision"),
            ]

            for m_title, m_attr in metric_configs:
                col = ctk.CTkFrame(dials_box, fg_color="transparent")
                col.pack(side="left", expand=True, fill="both")

                ctk.CTkLabel(
                    col, text=m_title,
                    font=OSFontResolver.get_mono(9, "bold"),
                    text_color=THEME["text_secondary"]
                ).pack(pady=(2, 4))

                for key in results.keys():
                    prof = BENCHMARK_PROFILES[key]
                    val = prof[m_attr]
                    dial = PrecisionRadialGauge(
                        col, value=val, size=88, thickness=7,
                        color=prof["color"],
                        title=prof["short_name"].split(" ")[0],
                        subtitle=f"{val:.1f}%",
                        bg_parent=THEME["bg_card"]
                    )
                    dial.pack(pady=2)
                    self.after(500, lambda dd=dial, vv=val: _safe_update_gauge(dd, vv))

        # SECTION 3: Structured 3-Column Agronomic Care Protocol Matrix
        ctk.CTkLabel(
            self.scroll_viewport, text="RECOMMENDED CARE GUIDELINES",
            font=OSFontResolver.get_mono(10, "bold"),
            text_color=THEME["text_muted"]
        ).pack(anchor="w", padx=6, pady=(16, 2))

        matrix_container = ctk.CTkFrame(
            self.scroll_viewport, fg_color=THEME["bg_panel"],
            corner_radius=4, border_width=1, border_color=THEME["border_hairline"]
        )
        matrix_container.pack(fill="x", padx=2, pady=(2, 14))

        # Top Threat Level Banner & Export Bar
        threat_type = protocol_data["threat_type"]
        if threat_type == "CRITICAL":
            threat_border = THEME["status_critical"]
            threat_bg = THEME["status_critical_bg"]
            threat_txt = THEME["status_critical_txt"]
        elif threat_type == "WARNING":
            threat_border = THEME["status_warning"]
            threat_bg = THEME["status_warning_bg"]
            threat_txt = THEME["status_warning_txt"]
        else:
            threat_border = THEME["status_nominal"]
            threat_bg = THEME["status_nominal_bg"]
            threat_txt = THEME["status_nominal_txt"]

        banner_header = ctk.CTkFrame(
            matrix_container, fg_color=THEME["bg_card_sub"],
            corner_radius=0, height=44
        )
        banner_header.pack(fill="x")
        banner_header.pack_propagate(False)

        badge_box = ctk.CTkFrame(
            banner_header, fg_color=threat_bg, corner_radius=3,
            border_width=1, border_color=threat_border
        )
        badge_box.pack(side="left", padx=10, pady=8)
        ctk.CTkLabel(
            badge_box, text=f"Threat Level: {protocol_data['threat_level']}",
            font=OSFontResolver.get_ui(9, "bold"),
            text_color=threat_txt
        ).pack(padx=8, pady=2)

        ctk.CTkLabel(
            banner_header, text=f"Condition: {protocol_data['condition']}",
            font=OSFontResolver.get_ui(10, "bold"),
            text_color=THEME["text_primary"]
        ).pack(side="left", padx=10)

        # Export Report Button
        export_btn = EnterpriseButton(
            banner_header, text="Save Report (.txt / .json)",
            font=OSFontResolver.get_ui(10, "bold"),
            height=28, fg_color=THEME["bg_input"],
            hover_color=THEME["border_hairline"],
            command=self._on_export_report
        )
        export_btn.pack(side="right", padx=10, pady=8)

        # 3 Structured Protocol Cards in Responsive Horizontal Grid
        cols_row = ctk.CTkFrame(matrix_container, fg_color="transparent")
        cols_row.pack(fill="x", padx=6, pady=8)

        cards_spec = [
            ("section_01", THEME["accent"]),
            ("section_02", THEME["status_warning_txt"]),
            ("section_03", THEME["status_nominal_txt"]),
        ]

        for sec_key, header_accent in cards_spec:
            sec_data = protocol_data["sections"][sec_key]

            col_card = ctk.CTkFrame(
                cols_row, fg_color=THEME["bg_card"], corner_radius=4,
                border_width=1, border_color=THEME["border_hairline"]
            )
            col_card.pack(side="left", expand=True, fill="both", padx=3, pady=2)

            # Card Header
            hdr_f = ctk.CTkFrame(col_card, fg_color=THEME["bg_card_sub"], corner_radius=0, height=32)
            hdr_f.pack(fill="x")
            hdr_f.pack_propagate(False)

            ctk.CTkLabel(
                hdr_f, text=sec_data["title"],
                font=OSFontResolver.get_ui(9, "bold"),
                text_color=header_accent
            ).pack(side="left", padx=8)

            # Protocol Items (Indexed Steps)
            prefix = sec_key.split("_")[1].lstrip("0")
            for idx_step, step_text in enumerate(sec_data["items"], start=1):
                item_f = ctk.CTkFrame(col_card, fg_color="transparent")
                item_f.pack(fill="x", padx=8, pady=4)

                idx_tag = f"{prefix}.{idx_step}"
                ctk.CTkLabel(
                    item_f, text=idx_tag,
                    font=OSFontResolver.get_mono(8, "bold"),
                    text_color=THEME["text_muted"], width=24, anchor="nw"
                ).pack(side="left", anchor="nw")

                ctk.CTkLabel(
                    item_f, text=step_text,
                    font=OSFontResolver.get_ui(10, "normal"),
                    text_color=THEME["text_secondary"],
                    wraplength=260, justify="left"
                ).pack(side="left", fill="x", expand=True, padx=(4, 0))

            ctk.CTkFrame(col_card, height=6, fg_color="transparent").pack()

    # --------------------------------------------------------------------------
    # Export Report (.TXT / .JSON)
    # --------------------------------------------------------------------------
    def _on_export_report(self):
        if not self._latest_report_data:
            messagebox.showinfo("Export Notice", "Please analyze an image before exporting a report.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Diagnostic Report",
            defaultextension=".txt",
            filetypes=[("Text Report (*.txt)", "*.txt"), ("JSON Object (*.json)", "*.json")]
        )
        if not file_path:
            return

        data = self._latest_report_data
        proto = data["protocol"]

        if file_path.endswith(".json"):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Report Saved", f"Diagnostic report saved successfully:\n{file_path}")
            return

        # Generate Human-Readable Text Document
        lines = [
            "================================================================================",
            "PHYTOSCAN · BOTANICAL PATHOLOGY & AGRONOMIC DIAGNOSTIC REPORT",
            "================================================================================",
            f"TIMESTAMP           : {data['timestamp']}",
            f"SPECIMEN FILE       : {data['specimen_path']}",
            f"HARDWARE PLATFORM   : {ENGINE.device_name}",
            f"SYSTEM VERSION      : v4.2",
            "--------------------------------------------------------------------------------",
            "DIAGNOSTIC FINDINGS:",
            f"Identified Condition: {proto['condition']}",
            f"Taxonomic Pathogen  : {proto['pathogen']}",
            f"Threat Level        : {proto['threat_level']}",
            "--------------------------------------------------------------------------------",
            "EVALUATED MODEL PREDICTIONS:",
        ]
        for m_k, preds in data["results"].items():
            if preds and "error" not in preds[0]:
                top = preds[0]
                prof = BENCHMARK_PROFILES[m_k]
                lines.append(f"  • {prof['short_name']:<24} : {top['plant']} - {top['diagnosis']} ({top['confidence_pct']:.2f}% Confidence)")

        lines.extend([
            "--------------------------------------------------------------------------------",
            "RECOMMENDED AGRONOMIC CARE & TREATMENT GUIDELINES:",
            "",
            proto["sections"]["section_01"]["title"],
        ])
        for i, s in enumerate(proto["sections"]["section_01"]["items"], 1):
            lines.append(f"  1.{i} {s}")

        lines.extend(["", proto["sections"]["section_02"]["title"]])
        for i, s in enumerate(proto["sections"]["section_02"]["items"], 1):
            lines.append(f"  2.{i} {s}")

        lines.extend(["", proto["sections"]["section_03"]["title"]])
        for i, s in enumerate(proto["sections"]["section_03"]["items"], 1):
            lines.append(f"  3.{i} {s}")

        lines.extend([
            "",
            "================================================================================",
            "CONFIDENTIAL SCIENTIFIC REPORT — FOR BOTANICAL & HORTICULTURAL REFERENCE",
            "================================================================================",
        ])

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        messagebox.showinfo("Report Saved", f"Diagnostic report saved successfully:\n{Path(file_path).name}")


# ==============================================================================
# 11. WORKSTATION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    app = PhytoScanApp()
    app.mainloop()
