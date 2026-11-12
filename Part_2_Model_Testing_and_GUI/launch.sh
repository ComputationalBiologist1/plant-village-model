#!/usr/bin/env bash
# ==============================================================================
# PHYTOSCAN · BOTANICAL PATHOLOGY WORKSTATION (v4.2)
# Unified Cross-Platform Shell Launcher
# Supported Platforms: macOS (Darwin), Linux (X11/Wayland), Windows (Git Bash/WSL/MSYS2)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "PHYTOSCAN · BOTANICAL PATHOLOGY WORKSTATION (v4.2)"
echo "Initializing Desktop Interface..."
echo "========================================================================"

# Candidate Python search hierarchy
CANDIDATE_RUNTIMES=(
    # Virtualenv / Poetry in script directory
    "$SCRIPT_DIR/.venv/bin/python3"
    "$SCRIPT_DIR/.venv/bin/python"
    "$SCRIPT_DIR/.venv/Scripts/python.exe"
    "$SCRIPT_DIR/venv/bin/python3"
    "$SCRIPT_DIR/venv/bin/python"
    "$SCRIPT_DIR/venv/Scripts/python.exe"
    # Conda environment if active
    "${CONDA_PREFIX}/bin/python3"
    "${CONDA_PREFIX}/bin/python"
    "${CONDA_PREFIX}/python.exe"
    # macOS system / framework Python installations
    "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
    "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
    "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"
    "/opt/homebrew/bin/python3"
    "/usr/local/bin/python3"
    # PATH discovery
    "$(which python3.11 2>/dev/null || true)"
    "$(which python3.10 2>/dev/null || true)"
    "$(which python3 2>/dev/null || true)"
    "$(which python 2>/dev/null || true)"
)

TARGET_PYTHON=""

# Verification Phase 1: Test for torch and customtkinter
for PY_CANDIDATE in "${CANDIDATE_RUNTIMES[@]}"; do
    if [ -n "$PY_CANDIDATE" ] && [ -x "$PY_CANDIDATE" ] 2>/dev/null; then
        if "$PY_CANDIDATE" -c "import torch, torchvision, PIL, customtkinter" 2>/dev/null; then
            TARGET_PYTHON="$PY_CANDIDATE"
            break
        fi
    fi
done

# Verification Phase 2: Fallback for torch and PIL
if [ -z "$TARGET_PYTHON" ]; then
    for PY_CANDIDATE in "${CANDIDATE_RUNTIMES[@]}"; do
        if [ -n "$PY_CANDIDATE" ] && [ -x "$PY_CANDIDATE" ] 2>/dev/null; then
            if "$PY_CANDIDATE" -c "import torch, PIL" 2>/dev/null; then
                TARGET_PYTHON="$PY_CANDIDATE"
                break
            fi
        fi
    done
fi

if [ -z "$TARGET_PYTHON" ]; then
    echo "[ERROR] Compatible Python runtime with PyTorch not detected."
    echo "Please set up your Python environment via:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

PY_VERSION="$("$TARGET_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
HOST_OS="$(uname -s 2>/dev/null || echo "Unknown-OS")"
HOST_ARCH="$(uname -m 2>/dev/null || echo "x86_64")"

echo "Host Environment : $HOST_OS ($HOST_ARCH)"
echo "Verified Runtime : $TARGET_PYTHON (v$PY_VERSION)"
echo "Interface        : Launching application..."
echo "========================================================================"

exec "$TARGET_PYTHON" "$SCRIPT_DIR/plantguard_app.py" "$@"
