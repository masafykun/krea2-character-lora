# LoRA weights

The trained LoRA checkpoints are hosted on Hugging Face (they are too large for
this Git repository and belong on a model host):

> **https://huggingface.co/masafy/krea2-character-lora**

| File | Epoch | Size | Notes |
|---|---|---|---|
| `masafee_krea2_e06.safetensors` | 6 | ~448 MB | retains eye catchlights, flexible backgrounds |
| `masafee_krea2_e08.safetensors` | 8 | ~448 MB | balanced |
| `masafee_krea2_e10.safetensors` | 10 | ~448 MB | strongest form; use at strength 0.8 + eye-highlight prompt |

**Recommended:** `e10` at LoRA strength **0.8**, with a prompt that includes
`big sparkling eyes with bright highlights` (see the report, §Results).

## Download

```bash
# via the helper script
bash ../scripts/download_lora.sh

# or with huggingface_hub
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('masafy/krea2-character-lora', local_dir='lora')"
```

## Usage (ComfyUI)

Place the chosen `.safetensors` in `ComfyUI/models/loras/`, load
`scripts/inference_workflow_comfyui.json`, and generate with the trigger token
`masafee`. The base model is Krea 2 Turbo (GGUF Q5_K_M); see NOTICE.md.
