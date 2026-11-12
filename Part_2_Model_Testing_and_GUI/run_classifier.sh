#!/usr/bin/env zsh
# Helper runner that automatically selects the Python environment with PyTorch installed

if [ -f "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3" ]; then
    PY_BIN="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
elif command -v python3 &>/dev/null; then
    PY_BIN="python3"
else
    echo "❌ Error: Python not found."
    exit 1
fi

"$PY_BIN" "$(dirname "$0")/classify_leaf.py" "$@"
