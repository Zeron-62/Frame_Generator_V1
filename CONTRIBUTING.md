# Contributing

Thanks for contributing to FrameForge In-Between.

## Before opening an issue

Please include:

- Windows version
- GPU model and VRAM
- ComfyUI version/build
- ComfyUI_IPAdapter_plus version
- FrameForge version
- model family and checkpoint name
- resolution, FPS, duration, steps, CFG, and seed
- the exact error message or relevant ComfyUI console output

Do not upload model weights or private/generated images unless you have the right to share them.

## Pull requests

Keep changes focused and document user-visible behavior in `CHANGELOG.md`.

Before submitting:

```bat
python -m py_compile app.py
```

and verify the application can still connect to a local ComfyUI server.
