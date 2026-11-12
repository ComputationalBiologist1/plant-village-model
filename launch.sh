#!/usr/bin/env bash
# ==============================================================================
# PHYTOSCAN · BOTANICAL PATHOLOGY WORKSTATION (v4.2)
# Root Shell Launcher -> Delegates to Part 2 Workstation
# Supported Platforms: macOS (Darwin), Linux (X11/Wayland), Windows (Git Bash/WSL)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/Part_2_Model_Testing_and_GUI/launch.sh" ]; then
    exec "$SCRIPT_DIR/Part_2_Model_Testing_and_GUI/launch.sh" "$@"
elif [ -f "$SCRIPT_DIR/Compatable_run/launch.sh" ]; then
    exec "$SCRIPT_DIR/Compatable_run/launch.sh" "$@"
else
    echo "[ERROR] Part_2_Model_Testing_and_GUI/launch.sh not found."
    exit 1
fi
