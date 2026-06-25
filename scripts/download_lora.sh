#!/bin/bash
set -e
REPO="masafy/krea2-character-lora"
DEST="$(cd "$(dirname "$0")/.." && pwd)/lora"
mkdir -p "$DEST"
if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" --local-dir "$DEST"
elif python -c "import huggingface_hub" 2>/dev/null; then
  python -c "from huggingface_hub import snapshot_download; snapshot_download('$REPO', local_dir='$DEST')"
else
  echo "Install: pip install huggingface_hub ; or download from https://huggingface.co/$REPO"
  exit 1
fi
echo "LoRA weights downloaded to: $DEST"
