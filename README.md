# Frame_Generator_V1

> **Local AI frame-by-frame image generation from a first frame, last frame, and action prompt — powered by ComfyUI.**

![Frame_Generator_V1 frame sequence](docs/images/frame-sequence.png)

### ✨ What it does

Give Frame_Generator_V1 a **first frame**, a **last frame**, and an **action prompt**. It generates the intermediate images as individual PNG frames that you can use in your animation or video workflow.

```text
 First Frame ─────┐
                  │
 Action Prompt ───┼──► Frame_Generator_V1 ───► AI In-Between PNG Frames
                  │
 Last Frame ──────┘
```

> **Not a video generator.** The core output is an image sequence: the first and last images are your endpoints, and the middle images are generated individually.

---

## 🌟 Features

| Feature | Description |
|---|---|
| 🖼️ First + Last Frame | Supply both endpoint images |
| ✍️ Action Prompt | Describe the motion or pose change |
| 🤖 AI In-Between Frames | Generate individual middle frames with diffusion |
| 🎛️ Generation Controls | Seed, steps, CFG, denoise, conditioning and consistency |
| 📐 Resolution Control | Choose the output image size |
| 🎞️ FPS + Duration | Automatically calculate the required frame count |
| 🔗 IP-Adapter Conditioning | Use endpoint images as visual references |
| 🧩 ComfyUI Integration | Runs through a local ComfyUI API |
| 🖥️ 8 GB VRAM Profile | Includes a conservative SD 1.5 starting profile |
| 📦 PNG Sequence Export | Export all frames or only generated in-between frames |
| 🔍 Preview Frames | Test selected transition positions before rendering a full sequence |

---

## 🖥️ Example Result

**Start image → AI-generated in-between frames → End image**

![Example frame sequence](docs/images/frame-sequence.png)

---

## 🧰 Requirements

- **Windows**
- **NVIDIA GPU**
- **8 GB VRAM recommended minimum** for the included SD 1.5 profile
- **32 GB system RAM recommended**
- **ComfyUI Desktop** or a local ComfyUI server
- Python dependencies listed in `requirements.txt`

---

## 🚀 Installation

### 1. Install and start ComfyUI

Install **ComfyUI Desktop** or run a local ComfyUI server.

Frame_Generator_V1 connects to ComfyUI through its local HTTP API, so the ComfyUI Desktop installation directory does **not** need to be configured in Frame_Generator_V1.

Default ComfyUI URL:

```text
http://127.0.0.1:8188
```

### 2. Install the IP-Adapter node

In ComfyUI Manager / Extensions, install:

```text
ComfyUI_IPAdapter_plus
```

Restart ComfyUI after installation.

### 3. Install the required models

Frame_Generator_V1 does not distribute model weights.

Install the required checkpoint, IP-Adapter, and CLIP Vision files in the model directories used by your ComfyUI installation.

See [`docs/MODELS.md`](docs/MODELS.md) for the exact files and sources.

### 4. Install Frame_Generator_V1

Open Command Prompt in the project folder:

```bat
install.bat
```

Then start the application:

```bat
run.bat
```

The web UI opens at:

```text
http://127.0.0.1:7860
```

### 5. Connect to ComfyUI

Open **Setup** in Frame_Generator_V1.

Keep the default URL unless you changed the ComfyUI port:

```text
http://127.0.0.1:8188
```

Click:

**Save & Test Connection → Refresh model lists**

---

## 🎬 How to Generate Frames

1. Upload the **First frame**.
2. Upload the **Last frame**.
3. Enter an **Action / motion prompt**.
4. Select the checkpoint, IP-Adapter model, and CLIP Vision model.
5. Set resolution, FPS, duration, seed, steps, CFG, conditioning and consistency.
6. Use **Preview One Frame** at 25%, 50% or 75% when available.
7. Generate the PNG sequence.

### Example prompt

```text
The girl slowly turns her head from left to right,
her hair gently moving in the wind,
natural pose progression,
consistent character, clothing, camera, and background.
```

For a **2-second sequence at 12 FPS**:

```text
24 total frames
22 AI-generated in-between frames
2 original endpoint frames
```

---

## 🎛️ Recommended 8 GB VRAM Starting Profile

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
| 8 GB VRAM mode | Enabled |

> Every in-between frame is a separate image-generation job. Larger resolutions, longer durations and higher frame counts increase generation time and memory use.

---

## 🧠 Current Conditioning Approach

The current prototype can combine:

- weighted first-frame and last-frame IP-Adapter embeddings
- an adaptive denoise schedule that is lower near endpoints and higher toward the middle
- a hidden endpoint guide used only as an img2img starting point
- an optional low-weight previous-frame reference for temporal continuity
- configurable transition curves

The hidden guide is **not exported as a frame**.

**No frame interpolation or cross-fade is used as the final output.**

---

## 📦 Output

A completed generation can produce:

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

Generated output, logs, temporary inputs, virtual environments and Python cache files are intentionally excluded from Git.

---

## 🧪 Development

Create a virtual environment manually when needed:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Run the application:

```bat
python app.py
```

Check syntax without launching the UI:

```bat
python -m py_compile app.py
```

---

## 🛠️ Project Status

**Experimental / active development**

The project focuses on controllable, image-by-image generation between two endpoint images. Visual consistency can vary with the checkpoint, prompt, source artwork and conditioning settings.

The repository is intentionally focused on **local image generation rather than direct video generation**.

---

## 🗺️ Roadmap

Planned refinement areas include:

- stronger character and background consistency
- improved motion distribution across the entire sequence
- more controllable temporal conditioning
- better handling of hands, hair and small moving details
- additional SD / SDXL model profiles
- improved previews and debugging
- easier workflow/model setup

---

## 🤝 Contributing

Contributions, experiments, workflow improvements and bug reports are welcome.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.

---

## 📄 License

The Frame_Generator_V1 project code is released under the **MIT License**.

See [`LICENSE`](LICENSE).

Third-party software, model weights and node packages remain under their own licenses and terms. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

---

## ⭐ About

Frame_Generator_V1 is a local AI experiment built around a simple idea:

> **Use AI to create the images between two keyframes — one frame at a time.**
