# Frame_Generator_V1

Local AI frame-by-frame image generation from a first frame, last frame, and action prompt using ComfyUI.

![Frame_Generator_V1 frame generation](docs/images/frame-sequence.png)

## What is Frame_Generator_V1?

Frame_Generator_V1 is a local AI tool for generating individual in-between PNG frames between two supplied images.

Instead of generating an MP4 directly, the application generates the intermediate frames as separate images that can be imported into video or animation software.

## Features

- First-frame and last-frame input
- Action/motion prompt
- AI-generated in-between PNG frames
- SD 1.5 support
- IP-Adapter conditioning
- Seed control
- Resolution control
- FPS and duration controls
- Adjustable denoise and guidance
- 8 GB VRAM mode
- ComfyUI integration
- PNG sequence export

## Requirements

- Windows
- NVIDIA GPU
- 8 GB VRAM recommended minimum for the included SD 1.5 profile
- 32 GB system RAM recommended
- ComfyUI Desktop
- Python dependencies listed in `requirements.txt`

## How It Works

```text
First Frame ─────┐
                 │
Action Prompt ───┼──> Frame_Generator_V1 ──> AI In-Between Frames
                 │
Last Frame ──────┘
The first and last images are preserved as the endpoints. The middle images are generated independently with diffusion, using both endpoint images as visual conditions.

## Current approach

Version 4 builds on the v3 prototype with a GitHub-ready project layout and clearer terminology around the frame-conditioning pipeline.

The generation pipeline can combine:

- weighted first-frame and last-frame IP-Adapter embeddings
- an adaptive denoise schedule that is lower near the endpoints and higher toward the middle
- a hidden endpoint guide used only as an img2img starting point
- an optional low-weight previous-frame reference for temporal continuity
- configurable transition curves

The hidden guide is **not exported as a frame**. No frame interpolation or cross-fade is used as the final output.

## Hardware target

The first profile is intended for an NVIDIA GPU around **8 GB VRAM** and **32 GB system RAM**.

Recommended starting point:

| Setting | Starting value |
|---|---:|
| Model family | SD 1.5 |
| Resolution | 640×360 |
| FPS | 12 |
| Duration | 2 s |
| Steps | 16 |
| CFG | 6 |
| Endpoint denoise | 0.36 |
| Middle denoise | 0.68 |
| Endpoint conditioning | 1.00 |
| Previous-frame consistency | 0.16 |
| Previous-frame guide anchor | 0.10 |
| Transition curve | Smoothstep |

A long sequence is expensive because **each in-between frame is a separate image-generation job**. The app limits a single job to 120 total frames.

## Installation

### 1. Install ComfyUI

Install and start ComfyUI Desktop (or run a local ComfyUI server).

FrameForge connects over the ComfyUI HTTP API, so **you do not need to configure the ComfyUI Desktop installation directory**.

Default URL:

```text
http://127.0.0.1:8188
```

### 2. Install the IP-Adapter node

In ComfyUI Manager/Extensions, install:

```text
ComfyUI_IPAdapter_plus
```

Restart ComfyUI after installation.

### 3. Install model files

Place the required model files into the model directories used by your ComfyUI installation. Typical paths are:

```text
ComfyUI/models/checkpoints/
ComfyUI/models/ipadapter/
ComfyUI/models/clip_vision/
```

See [`docs/MODELS.md`](docs/MODELS.md) for the exact files and official sources.

### 4. Install Frame_Generator_V1

Open Command Prompt in the project folder and run:

```bat
install.bat
```

Then start Frame_Generator_V1:

```bat
run.bat
```

The web UI opens on:

```text
http://127.0.0.1:7860
```

### 5. Connect Frame_Generator_V1 to ComfyUI

Open **Setup** in Frame_Generator_V1.

Keep:

```text
http://127.0.0.1:8188
```

Click **Save & Test Connection**, then **Refresh model lists**.

## Usage

1. Upload the **First frame**.
2. Upload the **Last frame**.
3. Write an **Action / motion prompt**.
4. Select your checkpoint, IP-Adapter model, and CLIP Vision model.
5. Set resolution, FPS, duration, seed, steps, CFG, conditioning, and consistency settings.
6. Use **Preview One Frame** at 25%, 50%, or 75% before rendering a full sequence.
7. Generate the PNG sequence.

### Example prompt

```text
The girl slowly turns her head from left to right,
her hair gently moving in the wind, natural pose progression,
consistent character, clothing, camera, and background.
```

For a 2-second sequence at 12 FPS:

```text
24 total frames
22 AI-generated in-between frames
2 original endpoint frames
```

## Output

A completed job creates:

```text
output/
├── FrameForge_<job>_contact_sheet.jpg
├── FrameForge_<job>_all_frames.zip
├── FrameForge_<job>_inbetween_only.zip
├── FrameForge_<job>_metadata.json
└── frames/<job>/
    ├── frame_0001.png
    ├── frame_0002.png
    ├── ...
    └── frame_0024.png
```

Runtime/output directories are ignored by Git and should not be committed.

## GitHub repository layout

```text
Frame_Generator_V1/
├── app.py
├── config.json
├── requirements.txt
├── install.bat
├── run.bat
├── install_ipadapter.bat
├── LICENSE
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── THIRD_PARTY_NOTICES.md
├── docs/
│   └── MODELS.md
├── workflows/
│   └── README.txt
└── .github/
    └── ISSUE_TEMPLATE/
```

Model weights, generated images, logs, virtual environments, and Python cache files are intentionally excluded from the repository.

## Development

Create a virtual environment using the supplied `install.bat`, or manually:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Run:

```bat
python app.py
```

Check syntax without launching the UI:

```bat
python -m py_compile app.py
```

## Project status

Frame_Generator_V1 is an **experimental local tool**. The core objective is controllable image-by-image keyframe generation from two endpoint images. Visual consistency can vary with the checkpoint, prompt, reference art, and conditioning settings.

This repository is intentionally focused on the local image-generation pipeline rather than video generation.

## License

The Frame_Generator_V1 project code is released under the MIT License. See [`LICENSE`](LICENSE).

Third-party software, model weights, and node packages remain under their own licenses and terms. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
