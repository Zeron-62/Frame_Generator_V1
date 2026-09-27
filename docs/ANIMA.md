# Anima Engine Setup

Frame_Generator_V1 v7 adds experimental Anima Base v1.0 support. ComfyUI currently provides native Anima model loading through `UNETLoader`, `CLIPLoader`, and `VAELoader`; the official workflow uses `anima-base-v1.0.safetensors`, `qwen_3_06b_base.safetensors`, and `qwen_image_vae.safetensors`.

## Required model files

```text
models/diffusion_models/anima-base-v1.0.safetensors
models/text_encoders/qwen_3_06b_base.safetensors
models/vae/qwen_image_vae.safetensors
models/ipadapter/ip_adapter.safetensors
```

The Anima IP-Adapter node used by this project is `ComfyUI-Anima_IP-Adapter` by LuciferTC9527. Its README documents `models/ipadapter/ip_adapter.safetensors` and a SigLIP2 encoder; the loader can auto-download the encoder.

## Install the custom node

Use ComfyUI Manager / Extensions to install:

```text
ComfyUI-Anima_IP-Adapter
```

Then restart ComfyUI.

## 8 GB starting profile

Use:

```text
Resolution: 640x360
Steps: 12–16
CFG: 4–5
Anima IP strength: 0.55–0.75
Anima IP CFG: 2.0
Reference size: 512
Endpoint hint mix: 0.10–0.20
```

These are Frame_Generator starting values for a constrained 8 GB GPU, not the official model defaults. The official Anima model documentation describes 30–50 steps and CFG 4–5 for general generation.

## Licensing

The Anima model weights are governed by their own license and are not bundled with Frame_Generator_V1. Review the model license before redistribution or commercial use. The Anima IP-Adapter custom node code is Apache-2.0 according to its repository; its model weights may have separate terms.
