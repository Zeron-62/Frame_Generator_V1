# Models

The repository does **not** bundle model weights. Download them separately from their official sources and keep them outside Git.

## SD 1.5 checkpoint

Official source:

https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors

Place the downloaded file in the checkpoint directory used by your ComfyUI installation, typically:

```text
ComfyUI/models/checkpoints/
```

## IP-Adapter Plus for SD 1.5

Official source:

https://huggingface.co/h94/IP-Adapter/blob/main/models/ip-adapter-plus_sd15.safetensors

Place it in:

```text
ComfyUI/models/ipadapter/
```

## CLIP Vision ViT-H

Use the ViT-H image encoder required by the SD 1.5 IP-Adapter configuration. The upstream IP-Adapter documentation identifies:

```text
CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
```

and the `ComfyUI/models/clip_vision/` directory.

Upstream documentation:

https://github.com/cubiq/ComfyUI_IPAdapter_plus

## Verify in ComfyUI

After installing the files and restarting ComfyUI, confirm that these loaders expose the expected files:

- `CheckpointLoaderSimple`
- `IPAdapterModelLoader`
- `CLIPVisionLoader`

Then return to FrameForge and use **Setup → Refresh model lists**.

## Licensing reminder

The FrameForge MIT License applies to the FrameForge code, not to these model weights. Read the license/model card for every checkpoint, adapter, and encoder you download.
