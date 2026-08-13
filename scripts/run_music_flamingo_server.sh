#!/usr/bin/env bash
# Create (once) an isolated venv for the Music Flamingo server and run it.
#
# Kept separate from Side-Step's .venv on purpose: Side-Step pins
# transformers<4.58, MusicFlamingoForConditionalGeneration needs newer.
# Sharing the venv would break the trainer's text encoder.
#
# Usage:
#   ./scripts/run_music_flamingo_server.sh nvidia/music-flamingo-2601-hf
#   ./scripts/run_music_flamingo_server.sh /path/to/local/model 8100
#
# Costs ~3GB of disk for a second torch. That is the price of not
# breaking the trainer.

set -euo pipefail

MODEL="${1:-nvidia/music-flamingo-2601-hf}"
PORT="${2:-8100}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${MF_VENV_DIR:-$HOME/.venvs/music-flamingo}"
TORCH_INDEX="${MF_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

if ! command -v uv >/dev/null 2>&1; then
    echo "[FAIL] uv not found on PATH."
    echo "       Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[..] Creating venv at $VENV_DIR"
    uv venv --python 3.11 "$VENV_DIR"

    echo "[..] Installing torch from $TORCH_INDEX"
    uv pip install --python "$VENV_DIR" --index-url "$TORCH_INDEX" torch

    echo "[..] Installing server dependencies"
    uv pip install --python "$VENV_DIR" -r "$SCRIPT_DIR/music_flamingo_requirements.txt"
else
    echo "[ok] Reusing venv at $VENV_DIR"
    echo "     (delete it to force a clean reinstall)"
fi

# Fail early with a clear message rather than deep in a traceback.
if ! "$VENV_DIR/bin/python" -c "from transformers import MusicFlamingoForConditionalGeneration" 2>/dev/null; then
    echo "[FAIL] This transformers build has no MusicFlamingoForConditionalGeneration."
    echo "       Upgrade inside the server venv only:"
    echo "         uv pip install --python $VENV_DIR -U transformers"
    exit 1
fi

echo "[ok] Starting Music Flamingo on 127.0.0.1:$PORT"
echo "     Point Side-Step's Music Flamingo URL at: http://127.0.0.1:$PORT"
exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/music_flamingo_server.py" \
    --model "$MODEL" --port "$PORT"
