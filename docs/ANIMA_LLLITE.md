# Anima LLLite Structure Control

Frame_Generator_V1 v8.2 uses ComfyUI's native **Anima LLLite** model-patch route for the recommended Anima mode.

## Required components

Base Anima files:

- `anima-base-v1.0.safetensors` → `models/diffusion_models/`
- `qwen_3_06b_base.safetensors` → `models/text_encoders/`
- `qwen_image_vae.safetensors` → `models/vae/`

Structure-control patch:

- `anima-lllite-any-test-like-v2.safetensors` → `models/model_patches/`

The current official ComfyUI workflow lists the v2 Any Control patch in `models/model_patches/` and uses the native `ModelPatchLoader` + `AnimaLLLiteApply` path. This is the route used by v8. See the official workflow source:

https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_anima_lllite_any_control_to_image.json

Model source:

https://huggingface.co/Comfy-Org/Anima-LLLite/blob/main/model_patches/anima-lllite-any-test-like-v2.safetensors

## Why v8 uses LLLite

The previous Anima IP-Adapter experiments showed large composition drift and unstable reference encoding on an 8 GB GPU. v8 therefore separates the jobs:

- the **previous known-good frame** is used as the structural control image;
- the control image is processed as Canny and inverted, following the current official workflow pattern;
- **Canny + inversion** converts the control image to the geometry-oriented signal used by the current LLLite workflow;
- **Anima LLLite** injects that control into the Anima model;
- the **action prompt** describes motion rather than re-describing the whole scene.

The temporary control image is never exported as a final frame. Each generated frame starts from a fresh latent, preventing failed frames from being recursively fed into the sampler latent.

## 8 GB notes

v8 generates one image at a time. The Anima LLLite model-patch path is also single-frame (T=1), which matches the architecture of Frame_Generator_V1. Keep the first tests at 640×360 and 16–20 steps.

## Licensing

The Anima-LLLite weights are distributed under their own model license. Do not bundle the model weights in this repository. The Frame_Generator_V1 code license is separate.
