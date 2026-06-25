# NOTICE

This repository contains a LoRA (Low-Rank Adaptation) adapter and related
materials. Please observe the following attributions and licenses.

## Base model (not redistributed here)

The LoRA was trained on **Krea 2 RAW** and is intended for inference on
**Krea 2 Turbo**. The Krea 2 model weights are **not** redistributed in this
repository. The LoRA adapter is a derivative work of Krea 2 and its use is
subject to the **Krea 2 Community License**. Obtain the base weights from the
official source:

- Krea 2 RAW: https://huggingface.co/krea/Krea-2-Raw
- Krea 2 Turbo: https://huggingface.co/krea/Krea-2-Turbo

## Auxiliary models (not redistributed here)

- Text encoder: Qwen3-VL-4B-Instruct (`Comfy-Org/Qwen3-VL`) — subject to its own license.
- Autoencoder: Qwen-Image VAE (`Comfy-Org/Qwen-Image_ComfyUI`) — subject to its own license.

## Tooling

- Training: Musubi Tuner (https://github.com/kohya-ss/musubi-tuner).
- Inference: ComfyUI (https://github.com/comfyanonymous/ComfyUI).

## Character and training data

The original character **"Masafee"** and the 13 training images in `dataset/`
are the intellectual property of the author (Masato Suzuki / Masafy). They are
included here for reproducibility of the study. Redistribution or commercial
use of the character or its likeness requires the author's permission.

## LoRA weights

The trained LoRA adapter weights (hosted on Hugging Face) embody adaptations of
Krea 2 and are released for research and personal use under the terms of the
Krea 2 Community License, with attribution to this work.
