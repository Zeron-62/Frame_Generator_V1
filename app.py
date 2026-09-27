from __future__ import annotations

import json
import os
import secrets
import shutil
import threading
import time
import uuid
import zipfile
import webbrowser
from pathlib import Path
from typing import Any

import gradio as gr
import requests
from PIL import Image, ImageOps, ImageDraw

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
OUTPUT_DIR = BASE_DIR / "output"
FRAMES_DIR = OUTPUT_DIR / "frames"
LOG_DIR = BASE_DIR / "logs"
TMP_DIR = BASE_DIR / "temp_inputs"
for p in (OUTPUT_DIR, FRAMES_DIR, LOG_DIR, TMP_DIR):
    p.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG: dict[str, Any] = {
    "comfyui_url": "http://127.0.0.1:8188",
    "comfyui_timeout": 30,
    "generation_timeout_seconds": 7200,
    "default_resolution": "640x360",
    "default_fps": 12,
    "default_duration": 2.0,
    "default_steps": 16,
    "default_cfg": 6.0,
    "default_denoise": 0.68,
    "default_denoise_start": 0.36,
    "default_denoise_peak": 0.68,
    "default_endpoint_strength": 1.0,
    "default_temporal_strength": 0.16,
    "default_previous_guide_strength": 0.10,
    "default_curve": "Smoothstep",
    "default_model_family": "SD 1.5",
    "low_vram": True,
    "use_latent_guide": True,
    "use_previous_frame": True,
}

STOP_EVENT = threading.Event()


def load_config() -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                merged.update(raw)
        except Exception:
            pass
    CONFIG_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


CONFIG = load_config()


def comfy_url() -> str:
    return str(CONFIG.get("comfyui_url", DEFAULT_CONFIG["comfyui_url"])).rstrip("/")


def get_json(url: str, timeout: int = 30) -> Any:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def post_json(url: str, payload: Any, timeout: int = 30) -> Any:
    response = requests.post(url, json=payload, timeout=timeout)
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:2000]}")
    return response.json()


def check_comfy() -> tuple[bool, str]:
    try:
        response = requests.get(f"{comfy_url()}/system_stats", timeout=CONFIG["comfyui_timeout"])
        response.raise_for_status()
        data = response.json()
        devices = data.get("devices", [])
        if devices:
            parts = []
            for device in devices:
                if isinstance(device, dict):
                    name = device.get("name", "GPU")
                    total = float(device.get("vram_total", 0)) / 1024**3
                    free = float(device.get("vram_free", 0)) / 1024**3
                    parts.append(f"{name} — {total:.1f} GB VRAM ({free:.1f} GB free)")
            return True, "Connected. " + "; ".join(parts)
        return True, "Connected to ComfyUI."
    except Exception as exc:
        return False, f"Cannot reach ComfyUI at {comfy_url()}: {exc}"


def object_info(node_name: str) -> dict[str, Any]:
    try:
        data = get_json(f"{comfy_url()}/object_info/{node_name}", CONFIG["comfyui_timeout"])
        return data.get(node_name, {}) if isinstance(data, dict) else {}
    except Exception:
        return {}


def node_available(node_name: str) -> bool:
    return bool(object_info(node_name))


def _node_model_choices(node_name: str, input_name: str) -> list[str]:
    node = object_info(node_name)
    inputs = node.get("input", {}) if isinstance(node, dict) else {}
    required = inputs.get("required", {}) if isinstance(inputs, dict) else {}
    spec = required.get(input_name) if isinstance(required, dict) else None
    if isinstance(spec, list) and spec and isinstance(spec[0], list):
        return [str(x) for x in spec[0]]
    return []


def scan_models() -> dict[str, list[str]]:
    return {
        "checkpoint": _node_model_choices("CheckpointLoaderSimple", "ckpt_name"),
        "ipadapter": _node_model_choices("IPAdapterModelLoader", "ipadapter_file"),
        "clip_vision": _node_model_choices("CLIPVisionLoader", "clip_name"),
    }


def guess_models(family: str) -> tuple[str, str, str]:
    models = scan_models()

    def first_matching(items: list[str], terms: tuple[str, ...]) -> str | None:
        lower = [(x.lower(), x) for x in items]
        for term in terms:
            for low, original in lower:
                if term in low:
                    return original
        return None

    if family == "SDXL":
        checkpoint = first_matching(models["checkpoint"], ("sdxl", "juggernaut", "realvis"))
        ip = first_matching(models["ipadapter"], ("sdxl", "plus_sdxl", "ip-adapter-plus_sdxl"))
        clip = first_matching(models["clip_vision"], ("vit-h", "laion2b", "clip-vit-h"))
    else:
        checkpoint = first_matching(models["checkpoint"], ("v1-5", "sd15", "1.5", "dreamshaper"))
        ip = first_matching(models["ipadapter"], ("plus_sd15", "ip-adapter_sd15", "sd15"))
        clip = first_matching(models["clip_vision"], ("vit-h", "laion2b", "clip-vit-h"))
    checkpoint = checkpoint or (models["checkpoint"][0] if models["checkpoint"] else "")
    ip = ip or (models["ipadapter"][0] if models["ipadapter"] else "")
    clip = clip or (models["clip_vision"][0] if models["clip_vision"] else "")
    return checkpoint, ip, clip


def refresh_model_lists(family: str = "SD 1.5"):
    data = scan_models()
    cp, ip, clip = guess_models(family)
    return (
        gr.update(choices=data["checkpoint"], value=cp or None),
        gr.update(choices=data["ipadapter"], value=ip or None),
        gr.update(choices=data["clip_vision"], value=clip or None),
    )


def sync_family(family: str):
    return refresh_model_lists(family)


def normalize_input(path: str | None, label: str) -> Path:
    if not path:
        raise ValueError(f"{label} is required.")
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"{label} file not found: {source}")
    target = TMP_DIR / f"{label.replace(' ', '_')}_{uuid.uuid4().hex[:10]}{source.suffix.lower() or '.png'}"
    shutil.copy2(source, target)
    return target


def prep_image(image_path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    return ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def make_morph_guide(first_path: Path, last_path: Path, t: float, width: int, height: int) -> Path:
    first = prep_image(first_path, width, height)
    last = prep_image(last_path, width, height)
    # This cross-fade is NEVER returned as a frame. It is only a latent initialization guide
    # so the diffusion process knows the intended spatial progression between endpoints.
    guide = Image.blend(first, last, float(max(0.0, min(1.0, t))))
    path = TMP_DIR / f"guide_{uuid.uuid4().hex[:10]}.png"
    guide.save(path, "PNG")
    return path

def make_progressive_guide(first_path: Path, last_path: Path, previous_path: Path | None, t: float, width: int, height: int, previous_strength: float) -> Path:
    """Create a hidden latent-start guide from the endpoint morph plus a small previous-frame anchor.

    The resulting guide is never exported. It only gives img2img a spatially plausible starting point.
    """
    first = prep_image(first_path, width, height)
    last = prep_image(last_path, width, height)
    guide = Image.blend(first, last, float(max(0.0, min(1.0, t))))
    if previous_path is not None and previous_strength > 0:
        prev = prep_image(previous_path, width, height)
        guide = Image.blend(guide, prev, float(max(0.0, min(0.35, previous_strength))))
    path = TMP_DIR / f"guide_{uuid.uuid4().hex[:10]}.png"
    guide.save(path, "PNG")
    return path


def upload_image(path: Path) -> str:
    with path.open("rb") as handle:
        files = {"image": (path.name, handle, "application/octet-stream")}
        data = {"overwrite": "true", "type": "input", "subfolder": "frameforge"}
        response = requests.post(
            f"{comfy_url()}/upload/image",
            files=files,
            data=data,
            timeout=CONFIG["comfyui_timeout"],
        )
    response.raise_for_status()
    result = response.json()
    name = result.get("name")
    if not name:
        raise RuntimeError(f"Unexpected upload response: {result}")
    subfolder = result.get("subfolder", "")
    return f"{subfolder}/{name}" if subfolder else name


def time_curve(t: float, curve: str) -> float:
    t = max(0.0, min(1.0, float(t)))
    if curve == "Smoothstep":
        return t * t * (3.0 - 2.0 * t)
    if curve == "Smootherstep":
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    if curve == "Fast middle":
        return t ** 0.72
    if curve == "Fast start":
        return 1.0 - (1.0 - t) ** 0.72
    return t

def adaptive_denoise(t: float, edge_denoise: float, peak_denoise: float, curve: str = "Smoothstep") -> float:
    """Use low denoise near either endpoint and stronger denoise around the middle.

    This reduces the abrupt pose jump seen in early/late frames while allowing more freedom in the middle.
    """
    edge = 1.0 - abs(2.0 * max(0.0, min(1.0, t)) - 1.0)
    shaped = time_curve(edge, curve)
    return float(edge_denoise + (peak_denoise - edge_denoise) * shaped)


def endpoint_weights(t: float, total_strength: float, temporal_strength: float, curve: str, has_previous: bool) -> tuple[float, float, float]:
    u = time_curve(t, curve)
    if not has_previous or temporal_strength <= 0:
        return total_strength * (1.0 - u), total_strength * u, 0.0
    prev = min(float(temporal_strength), max(0.0, float(total_strength) * 0.40))
    remain = max(0.0, float(total_strength) - prev)
    return remain * (1.0 - u), remain * u, prev


def build_modern_graph(
    checkpoint: str,
    ipadapter_file: str,
    clip_vision_file: str,
    first_image: str,
    last_image: str,
    previous_image: str | None,
    guide_image: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
    first_weight: float,
    last_weight: float,
    previous_weight: float,
) -> dict[str, dict[str, Any]]:
    """Modern IPAdapterEncoder + CombineEmbeds graph.

    Endpoint embeddings are weighted BEFORE combination, then applied once to the diffusion model.
    This avoids the old sequential two-IPAdapter stacking behavior.
    """
    graph: dict[str, dict[str, Any]] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": first_image}},
        "5": {"class_type": "LoadImage", "inputs": {"image": last_image}},
        "6": {"class_type": "LoadImage", "inputs": {"image": guide_image}},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": ipadapter_file}},
        "9": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": clip_vision_file}},
        "10": {"class_type": "IPAdapterEncoder", "inputs": {"ipadapter": ["8", 0], "image": ["4", 0], "weight": float(first_weight), "clip_vision": ["9", 0]}},
        "11": {"class_type": "IPAdapterEncoder", "inputs": {"ipadapter": ["8", 0], "image": ["5", 0], "weight": float(last_weight), "clip_vision": ["9", 0]}},
        "12": {"class_type": "IPAdapterCombineEmbeds", "inputs": {"embed1": ["10", 0], "embed2": ["11", 0], "method": "add"}},
        "13": {"class_type": "IPAdapterCombineEmbeds", "inputs": {"embed1": ["10", 1], "embed2": ["11", 1], "method": "add"}},
    }
    embed_source = "12"
    neg_source = "13"
    next_id = 14
    if previous_image and previous_weight > 0.0:
        graph[str(next_id)] = {"class_type": "LoadImage", "inputs": {"image": previous_image}}
        prev_load = str(next_id)
        next_id += 1
        graph[str(next_id)] = {
            "class_type": "IPAdapterEncoder",
            "inputs": {"ipadapter": ["8", 0], "image": [prev_load, 0], "weight": float(previous_weight), "clip_vision": ["9", 0]},
        }
        prev_enc = str(next_id)
        next_id += 1
        graph[str(next_id)] = {
            "class_type": "IPAdapterCombineEmbeds",
            "inputs": {"embed1": [embed_source, 0], "embed2": [prev_enc, 0], "method": "add"},
        }
        embed_source = str(next_id)
        next_id += 1
        graph[str(next_id)] = {
            "class_type": "IPAdapterCombineEmbeds",
            "inputs": {"embed1": [neg_source, 0], "embed2": [prev_enc, 1], "method": "add"},
        }
        neg_source = str(next_id)
        next_id += 1

    graph[str(next_id)] = {
        "class_type": "IPAdapterEmbeds",
        "inputs": {
            "model": ["1", 0],
            "ipadapter": ["8", 0],
            "pos_embed": [embed_source, 0],
            "weight": 1.0,
            "weight_type": "linear",
            "start_at": 0.0,
            "end_at": 1.0,
            "embeds_scaling": "V only",
            "neg_embed": [neg_source, 0],
            "clip_vision": ["9", 0],
        },
    }
    ipa_model = str(next_id)
    next_id += 1
    graph[str(next_id)] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": float(denoise),
            "model": [ipa_model, 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["7", 0],
        },
    }
    sampler = str(next_id)
    next_id += 1
    graph[str(next_id)] = {"class_type": "VAEDecode", "inputs": {"samples": [sampler, 0], "vae": ["1", 2]}}
    decoded = str(next_id)
    next_id += 1
    graph[str(next_id)] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "FrameForge", "images": [decoded, 0]},
    }
    return graph


def build_legacy_graph(
    checkpoint: str,
    ipadapter_file: str,
    clip_vision_file: str,
    first_image: str,
    last_image: str,
    guide_image: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
    first_weight: float,
    last_weight: float,
) -> dict[str, dict[str, Any]]:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": guide_image}},
        "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["1", 2]}},
        "6": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": ipadapter_file}},
        "7": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": clip_vision_file}},
        "8": {"class_type": "LoadImage", "inputs": {"image": first_image}},
        "9": {"class_type": "LoadImage", "inputs": {"image": last_image}},
        "10": {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": ["1", 0], "ipadapter": ["6", 0], "clip_vision": ["7", 0],
                "image": ["8", 0], "weight": float(first_weight), "weight_type": "linear",
                "combine_embeds": "average", "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only",
            },
        },
        "11": {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": ["10", 0], "ipadapter": ["6", 0], "clip_vision": ["7", 0],
                "image": ["9", 0], "weight": float(last_weight), "weight_type": "linear",
                "combine_embeds": "average", "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only",
            },
        },
        "12": {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(seed), "steps": int(steps), "cfg": float(cfg), "sampler_name": "euler",
                "scheduler": "normal", "denoise": float(denoise), "model": ["11", 0],
                "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["5", 0],
            },
        },
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}},
        "14": {"class_type": "SaveImage", "inputs": {"filename_prefix": "FrameForge", "images": ["13", 0]}},
    }


def api_graph(
    checkpoint: str,
    ipadapter_file: str,
    clip_vision_file: str,
    first_image: str,
    last_image: str,
    previous_image: str | None,
    guide_image: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
    first_weight: float,
    last_weight: float,
    previous_weight: float,
) -> dict[str, dict[str, Any]]:
    modern = all(node_available(n) for n in ("IPAdapterEncoder", "IPAdapterCombineEmbeds", "IPAdapterEmbeds"))
    if modern:
        return build_modern_graph(
            checkpoint, ipadapter_file, clip_vision_file, first_image, last_image,
            previous_image, guide_image, prompt, negative_prompt, width, height,
            seed, steps, cfg, denoise, first_weight, last_weight, previous_weight,
        )
    return build_legacy_graph(
        checkpoint, ipadapter_file, clip_vision_file, first_image, last_image, guide_image,
        prompt, negative_prompt, width, height, seed, steps, cfg, denoise,
        first_weight, last_weight,
    )


def submit(graph: dict[str, Any]) -> str:
    payload = {"prompt": graph, "client_id": f"frameforge-{uuid.uuid4().hex}"}
    result = post_json(f"{comfy_url()}/prompt", payload, timeout=CONFIG["comfyui_timeout"])
    if result.get("node_errors"):
        raise RuntimeError(json.dumps(result["node_errors"], indent=2)[:6000])
    prompt_id = result.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"No prompt_id returned by ComfyUI: {result}")
    return str(prompt_id)


def find_images(history: dict[str, Any]) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []

    def walk(obj: Any):
        if isinstance(obj, dict):
            if isinstance(obj.get("filename"), str) and Path(obj["filename"]).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                found.append({
                    "filename": obj["filename"],
                    "subfolder": str(obj.get("subfolder", "")),
                    "type": str(obj.get("type", "output")),
                })
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(history)
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in found:
        key = (item["filename"], item["subfolder"], item["type"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def download_image(item: dict[str, str], destination: Path):
    params = {"filename": item["filename"], "subfolder": item["subfolder"], "type": item["type"]}
    response = requests.get(f"{comfy_url()}/view", params=params, timeout=CONFIG["comfyui_timeout"])
    response.raise_for_status()
    destination.write_bytes(response.content)


def interrupt_comfy() -> None:
    try:
        requests.post(f"{comfy_url()}/interrupt", timeout=5)
    except Exception:
        pass


def wait_for_image(prompt_id: str, progress_cb=None) -> Path:
    deadline = time.time() + int(CONFIG["generation_timeout_seconds"])
    while time.time() < deadline:
        if STOP_EVENT.is_set():
            interrupt_comfy()
            raise RuntimeError("Generation stopped by user.")
        try:
            history = get_json(f"{comfy_url()}/history/{prompt_id}", CONFIG["comfyui_timeout"])
            entry = history.get(prompt_id)
            if entry:
                status = entry.get("status", {}) if isinstance(entry, dict) else {}
                if isinstance(status, dict) and status.get("status_str") == "error":
                    raise RuntimeError(json.dumps(status, indent=2)[:6000])
                outputs = find_images(entry)
                if outputs:
                    local = TMP_DIR / f"{prompt_id}.png"
                    download_image(outputs[-1], local)
                    return local
        except requests.RequestException:
            pass
        if progress_cb:
            progress_cb()
        time.sleep(0.8)
    raise TimeoutError(f"Timed out waiting for ComfyUI job {prompt_id}.")


def save_contact_sheet(frame_paths: list[Path], destination: Path, columns: int = 6, tile=(240, 135)):
    rows = (len(frame_paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile[0], rows * (tile[1] + 24)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, path in enumerate(frame_paths):
        x = (idx % columns) * tile[0]
        y = (idx // columns) * (tile[1] + 24)
        try:
            image = Image.open(path).convert("RGB")
            image = ImageOps.fit(image, tile)
            sheet.paste(image, (x, y))
            draw.text((x + 6, y + tile[1] + 4), path.stem, fill="black")
        except Exception:
            continue
    sheet.save(destination, quality=94)


def zip_frames(frame_paths: list[Path], destination: Path):
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in frame_paths:
            zf.write(p, arcname=p.name)


def clean_sequence(seq_dir: Path):
    if seq_dir.exists():
        shutil.rmtree(seq_dir, ignore_errors=True)


def _validate_common(
    first: str | None, last: str | None, action_prompt: str,
    checkpoint: str, ipadapter: str, clip_vision: str,
) -> tuple[Path, Path]:
    if not first or not last:
        raise gr.Error("Upload both a first frame and a last frame.")
    if not action_prompt.strip():
        raise gr.Error("Enter an action / motion prompt.")
    if not checkpoint or not ipadapter or not clip_vision:
        raise gr.Error("Select a checkpoint, IP-Adapter model, and CLIP Vision model. Use Setup → Refresh model lists.")
    ok, status = check_comfy()
    if not ok:
        raise gr.Error(status)
    return normalize_input(first, "first_frame"), normalize_input(last, "last_frame")


def generation_prompt(action_prompt: str, t: float, style_notes: str) -> str:
    extra = style_notes.strip()
    base = (
        f"{action_prompt.strip()}. This is animation in-between frame {t*100:.1f}% through the action. "
        "Create a plausible intermediate state, not a frozen copy of either endpoint. "
        "Preserve the same character identity, face, hairstyle, clothing, camera angle, framing, background, "
        "lighting, art style and important objects. Only change what is physically implied by the requested motion. "
        "Maintain clean anatomy and coherent hands, eyes and limbs."
    )
    return base + (f" Additional consistency instructions: {extra}." if extra else "")


def generate_one(
    first: str | None,
    last: str | None,
    action_prompt: str,
    negative_prompt: str,
    style_notes: str,
    model_family: str,
    checkpoint: str,
    ipadapter: str,
    clip_vision: str,
    resolution: str,
    position: float,
    seed: float,
    steps: int,
    cfg: float,
    denoise_start: float,
    denoise_peak: float,
    endpoint_strength: float,
    low_vram: bool,
    use_latent_guide: bool,
    progress=gr.Progress(track_tqdm=False),
):
    STOP_EVENT.clear()
    try:
        first_path, last_path = _validate_common(first, last, action_prompt, checkpoint, ipadapter, clip_vision)
        width, height = (int(x) for x in resolution.lower().split("x"))
        if float(denoise_peak) < float(denoise_start):
            raise gr.Error("Middle denoise must be greater than or equal to endpoint denoise.")
        t = max(0.001, min(0.999, float(position) / 100.0))
        base_seed = secrets.randbelow(2**31 - 1) if int(seed) < 0 else int(seed)
        frame_denoise = adaptive_denoise(t, float(denoise_start), float(denoise_peak), "Smoothstep")
        first_w, last_w, _ = endpoint_weights(t, endpoint_strength, 0.0, "Linear", False)
        if use_latent_guide:
            guide_path = make_morph_guide(first_path, last_path, t, width, height)
        else:
            guide_image = prep_image(first_path, width, height)
            guide_path = TMP_DIR / f"pure_first_{uuid.uuid4().hex[:10]}.png"
            guide_image.save(guide_path, "PNG")
        first_ref = upload_image(first_path)
        last_ref = upload_image(last_path)
        guide_ref = upload_image(guide_path)
        graph = api_graph(
            checkpoint, ipadapter, clip_vision, first_ref, last_ref, None, guide_ref,
            generation_prompt(action_prompt, t, style_notes), negative_prompt.strip(),
            width, height, base_seed, steps, cfg, frame_denoise, first_w, last_w, 0.0,
        )
        progress(0.1, desc="Submitting preview frame…")
        pid = submit(graph)
        frame = wait_for_image(pid, lambda: progress(0.25, desc="Generating preview frame…"))
        out = OUTPUT_DIR / f"FrameForge_preview_{uuid.uuid4().hex[:10]}.png"
        shutil.copy2(frame, out)
        progress(1.0, desc="Preview complete")
        return str(out), f"Preview position: {position:.1f}%\nSeed: {base_seed}\nModel: {model_family}\nResolution: {width}x{height}\nAdaptive denoise: {frame_denoise:.3f}"
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def generate_sequence(
    first: str | None,
    last: str | None,
    action_prompt: str,
    negative_prompt: str,
    style_notes: str,
    model_family: str,
    checkpoint: str,
    ipadapter: str,
    clip_vision: str,
    resolution: str,
    fps: int,
    duration: float,
    seed: float,
    seed_mode: str,
    steps: int,
    cfg: float,
    denoise_start: float,
    denoise_peak: float,
    endpoint_strength: float,
    temporal_strength: float,
    previous_guide_strength: float,
    curve: str,
    first_lock: float,
    last_lock: float,
    low_vram: bool,
    use_latent_guide: bool,
    use_previous_frame: bool,
    progress=gr.Progress(track_tqdm=False),
):
    STOP_EVENT.clear()
    first_path, last_path = _validate_common(first, last, action_prompt, checkpoint, ipadapter, clip_vision)
    try:
        width, height = (int(x) for x in resolution.lower().split("x"))
    except Exception as exc:
        raise gr.Error(f"Invalid resolution: {resolution}") from exc
    if float(denoise_peak) < float(denoise_start):
        raise gr.Error("Middle denoise must be greater than or equal to endpoint denoise.")

    total_frames = max(3, int(round(float(fps) * float(duration))))
    if total_frames > 120:
        raise gr.Error("This build caps a single job at 120 total frames to protect 8 GB VRAM / 32 GB RAM systems.")
    middle_count = total_frames - 2

    try:
        base_seed = secrets.randbelow(2**31 - 1) if int(seed) < 0 else int(seed)
    except Exception as exc:
        raise gr.Error("Seed must be an integer or -1 for random.") from exc

    job_id = uuid.uuid4().hex[:10]
    render_dir = FRAMES_DIR / job_id
    clean_sequence(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    first_out = render_dir / "frame_0001.png"
    last_out = render_dir / f"frame_{total_frames:04d}.png"
    shutil.copy2(first_path, first_out)
    shutil.copy2(last_path, last_out)

    first_ref = upload_image(first_path)
    last_ref = upload_image(last_path)
    generated: list[Path] = []
    previous_path: Path | None = None

    progress(0.03, desc=f"Preparing {total_frames} frames…")

    try:
        for n in range(1, total_frames - 1):
            if STOP_EVENT.is_set():
                interrupt_comfy()
                raise RuntimeError("Generation stopped by user.")
            t = n / (total_frames - 1)
            frame_denoise = adaptive_denoise(t, float(denoise_start), float(denoise_peak), curve)
            edge_progress = 1.0 - abs(2.0 * t - 1.0)
            temporal_factor = 0.40 + 0.60 * time_curve(edge_progress, curve)
            effective_temporal = float(temporal_strength) * temporal_factor if use_previous_frame else 0.0
            first_w, last_w, prev_w = endpoint_weights(
                t, float(endpoint_strength), effective_temporal, curve, previous_path is not None,
            )
            # Endpoint locks are optional shaping curves applied without changing the total conditioning budget.
            first_w *= max(0.35, min(1.65, 1.20 / max(0.5, float(first_lock))))
            last_w *= max(0.35, min(1.65, 1.20 / max(0.5, float(last_lock))))
            # Renormalize endpoint pair so the conditioning budget remains stable after lock shaping.
            target = max(0.0001, (first_w + last_w))
            if prev_w <= 0:
                scale = float(endpoint_strength) / target
                first_w *= scale
                last_w *= scale
            else:
                remain = max(0.0001, float(endpoint_strength) - prev_w)
                scale = remain / target
                first_w *= scale
                last_w *= scale

            frame_seed = base_seed if seed_mode == "Fixed" else (base_seed + n)
            guide_ref: str | None = None
            guide_path: Path | None = None
            if use_latent_guide:
                guide_anchor = float(previous_guide_strength) if (use_previous_frame and previous_path is not None) else 0.0
                guide_path = make_progressive_guide(first_path, last_path, previous_path, t, width, height, guide_anchor)
            else:
                # A first-frame initialization is still used as a stable latent; the image is generated from noise.
                temp = prep_image(first_path, width, height)
                guide_path = TMP_DIR / f"guide_{uuid.uuid4().hex[:10]}.png"
                temp.save(guide_path, "PNG")
            guide_ref = upload_image(guide_path)
            prev_ref = upload_image(previous_path) if previous_path is not None and prev_w > 0 else None
            stage_prompt = generation_prompt(action_prompt, t, style_notes)
            graph = api_graph(
                checkpoint, ipadapter, clip_vision, first_ref, last_ref, prev_ref, guide_ref,
                stage_prompt, negative_prompt.strip(), width, height, int(frame_seed), int(steps),
                float(cfg), float(frame_denoise), float(first_w), float(last_w), float(prev_w),
            )
            pid = submit(graph)
            current_index = n + 1

            def cb():
                progress(
                    0.05 + 0.90 * ((n - 1) / max(1, middle_count)),
                    desc=f"Generating frame {current_index}/{total_frames} — {t*100:.1f}% through action…",
                )

            tmp = wait_for_image(pid, progress_cb=cb)
            destination = render_dir / f"frame_{current_index:04d}.png"
            shutil.copy2(tmp, destination)
            generated.append(destination)
            previous_path = destination

        all_frames = [first_out] + generated + [last_out]
        all_zip = OUTPUT_DIR / f"FrameForge_{job_id}_all_frames.zip"
        middle_zip = OUTPUT_DIR / f"FrameForge_{job_id}_inbetween_only.zip"
        contact = OUTPUT_DIR / f"FrameForge_{job_id}_contact_sheet.jpg"
        metadata = OUTPUT_DIR / f"FrameForge_{job_id}_metadata.json"
        zip_frames(all_frames, all_zip)
        zip_frames(generated, middle_zip)
        save_contact_sheet(all_frames, contact)
        metadata.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "model_family": model_family,
                    "checkpoint": checkpoint,
                    "ipadapter": ipadapter,
                    "clip_vision": clip_vision,
                    "total_frames": total_frames,
                    "ai_inbetween_frames": middle_count,
                    "fps": fps,
                    "duration_seconds": float(duration),
                    "resolution": [width, height],
                    "seed": base_seed,
                    "seed_mode": seed_mode,
                    "steps": steps,
                    "cfg": cfg,
                    "denoise_start": float(denoise_start),
                    "denoise_peak": float(denoise_peak),
                    "endpoint_strength": endpoint_strength,
                    "temporal_strength": temporal_strength if use_previous_frame else 0.0,
                    "previous_guide_strength": previous_guide_strength if use_previous_frame else 0.0,
                    "transition_curve": curve,
                    "latent_guide": bool(use_latent_guide),
                    "previous_frame_reference": bool(use_previous_frame),
                    "negative_prompt": negative_prompt,
                    "action_prompt": action_prompt,
                    "consistency_notes": style_notes,
                    "method": "weighted endpoint embeddings + adaptive denoise schedule + hidden progressive endpoint morph latent guide + optional previous-frame IPAdapter/reference; PNG frames only",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        log_text = (
            f"Job: {job_id}\n"
            f"Total frames: {total_frames}\n"
            f"AI-generated middle frames: {middle_count}\n"
            f"FPS / duration: {fps} / {duration:.2f}s\n"
            f"Resolution: {width}x{height}\n"
            f"Seed: {base_seed} ({seed_mode})\n"
            f"Endpoint strength: {endpoint_strength:.2f}\n"
            f"Denoise schedule: {denoise_start:.2f} → {denoise_peak:.2f} (adaptive, low at endpoints / high in middle)\n"
            f"Temporal reference: {temporal_strength:.2f} ({'ON' if use_previous_frame else 'OFF'})\n"
            f"Previous guide anchor: {previous_guide_strength:.2f} ({'ON' if use_previous_frame else 'OFF'})\n"
            f"Curve: {curve}\n"
            f"Latent guide: {'ON' if use_latent_guide else 'OFF'}\n"
            f"Method: weighted endpoint embeddings + adaptive denoise + progressive hidden latent guide + optional previous-frame reference.\n"
            f"No MP4/video export is generated.\n"
            f"Output folder: {render_dir}\n"
        )
        progress(1.0, desc="Done — PNG frame sequence created")
        return str(contact), str(all_zip), str(middle_zip), str(metadata), log_text
    except gr.Error:
        raise
    except Exception as exc:
        progress(1.0, desc="Generation stopped or failed")
        raise gr.Error(str(exc)) from exc


def save_comfy_url(url: str):
    url = str(url or "").strip().rstrip("/")
    if not url:
        return "Enter a ComfyUI URL."
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "http://" + url
    CONFIG["comfyui_url"] = url
    CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
    ok, status = check_comfy()
    return status if ok else f"Saved URL. {status}"


def open_comfy():
    webbrowser.open(comfy_url())
    return f"Opened {comfy_url()}"


def stop_generation():
    STOP_EVENT.set()
    interrupt_comfy()
    return "Stop requested. The active ComfyUI job will be interrupted."


def count_frames(fps: float, duration: float):
    total = max(3, int(round(float(fps) * float(duration))))
    return f"**{total} total frames** → **{total-2} AI in-between frames** (first + last are kept unchanged)"


def capabilities_text() -> str:
    nodes = ["IPAdapterEncoder", "IPAdapterCombineEmbeds", "IPAdapterEmbeds"]
    state = {n: node_available(n) for n in nodes}
    if all(state.values()):
        return "✅ Modern weighted-embedding workflow available. FrameForge combines endpoint embeddings, uses adaptive denoise, and can anchor each frame to the previous generated frame."
    missing = ", ".join(n for n, ok in state.items() if not ok)
    return f"⚠️ Modern embedding workflow missing: {missing}. FrameForge will use the compatibility workflow."


def app():
    models = scan_models()
    family_default = str(CONFIG.get("default_model_family", "SD 1.5"))
    cp, ip, clip = guess_models(family_default)
    resolutions = ["512x512", "640x360", "640x384", "768x432", "768x512", "832x480", "1024x576", "1024x1024"]

    with gr.Blocks(title="FrameForge In-Between") as demo:
        gr.Markdown(
            "# 🖼️ FrameForge In-Between — v4\n"
            "### First frame + last frame + action → **AI-generated PNG in-between frames**\n\n"
            "This is a frame generator, not a video generator. The middle frames are generated as individual diffusion images."
        )

        with gr.Tab("Generate Frames"):
            with gr.Row():
                with gr.Column(scale=1):
                    first = gr.Image(type="filepath", sources=["upload", "clipboard"], label="First frame")
                    last = gr.Image(type="filepath", sources=["upload", "clipboard"], label="Last frame")
                with gr.Column(scale=1):
                    action_prompt = gr.Textbox(
                        label="Action / motion prompt",
                        lines=6,
                        placeholder="Example: the girl slowly turns her head from left to right, raises her right hand, hair moves gently in the wind",
                    )
                    style_notes = gr.Textbox(
                        label="Consistency instructions (optional)",
                        lines=3,
                        value="same character design, same outfit, same camera and background, preserve anime line art and color palette",
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative prompt",
                        lines=3,
                        value="deformed hands, extra fingers, extra limbs, duplicate character, inconsistent face, changed clothing, changed background, warped anatomy, text, watermark, blurry, low quality",
                    )
                    model_family = gr.Radio(["SD 1.5", "SDXL"], value=family_default if family_default in {"SD 1.5", "SDXL"} else "SD 1.5", label="Model family")

            with gr.Row():
                checkpoint = gr.Dropdown(choices=models["checkpoint"], value=cp or None, label="Checkpoint")
                ipadapter = gr.Dropdown(choices=models["ipadapter"], value=ip or None, label="IP-Adapter model")
                clip_vision = gr.Dropdown(choices=models["clip_vision"], value=clip or None, label="CLIP Vision")

            with gr.Accordion("Frame timing", open=True):
                with gr.Row():
                    resolution = gr.Dropdown(resolutions, value=str(CONFIG.get("default_resolution", "640x360")), label="Resolution")
                    fps = gr.Slider(6, 30, step=1, value=float(CONFIG.get("default_fps", 12)), label="FPS")
                    duration = gr.Slider(0.5, 10, step=0.5, value=float(CONFIG.get("default_duration", 2.0)), label="Duration (s)")
                frame_count = gr.Markdown(count_frames(CONFIG.get("default_fps", 12), CONFIG.get("default_duration", 2.0)))
                fps.change(count_frames, inputs=[fps, duration], outputs=frame_count)
                duration.change(count_frames, inputs=[fps, duration], outputs=frame_count)

            with gr.Accordion("AI generation", open=True):
                with gr.Row():
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    seed_mode = gr.Radio(["Fixed", "Increment"], value="Fixed", label="Seed strategy")
                    steps = gr.Slider(6, 40, step=1, value=int(CONFIG.get("default_steps", 16)), label="Steps")
                    cfg = gr.Slider(1, 12, step=0.5, value=float(CONFIG.get("default_cfg", 6.0)), label="CFG")
                with gr.Row():
                    denoise_start = gr.Slider(0.20, 0.60, step=0.01, value=float(CONFIG.get("default_denoise_start", 0.36)), label="Endpoint denoise (preserve endpoints)")
                    denoise_peak = gr.Slider(0.45, 0.90, step=0.01, value=float(CONFIG.get("default_denoise_peak", 0.68)), label="Middle denoise (motion freedom)")
                    endpoint_strength = gr.Slider(0.40, 1.50, step=0.05, value=float(CONFIG.get("default_endpoint_strength", 1.0)), label="Endpoint conditioning budget")
                    temporal_strength = gr.Slider(0.0, 0.35, step=0.01, value=float(CONFIG.get("default_temporal_strength", 0.16)), label="Previous-frame consistency")
                with gr.Row():
                    previous_guide_strength = gr.Slider(0.0, 0.30, step=0.01, value=float(CONFIG.get("default_previous_guide_strength", 0.10)), label="Previous-frame guide anchor")
                    curve = gr.Dropdown(["Linear", "Smoothstep", "Smootherstep", "Fast middle", "Fast start"], value=str(CONFIG.get("default_curve", "Smoothstep")), label="Transition curve")
                with gr.Row():
                    first_lock = gr.Slider(0.5, 2.5, step=0.1, value=1.0, label="First endpoint lock")
                    last_lock = gr.Slider(0.5, 2.5, step=0.1, value=1.0, label="Last endpoint lock")
                with gr.Row():
                    low_vram = gr.Checkbox(value=bool(CONFIG.get("low_vram", True)), label="8 GB VRAM profile (guidance)")
                    use_latent_guide = gr.Checkbox(value=bool(CONFIG.get("use_latent_guide", True)), label="Hidden endpoint morph guide (recommended)")
                    use_previous_frame = gr.Checkbox(value=bool(CONFIG.get("use_previous_frame", True)), label="Use previous generated frame")
                gr.Markdown(
                    "The hidden guide is **never exported**. Denoise is now adaptive: low near the endpoints and higher in the middle. "
                    "The previous-frame reference and guide anchor are intentionally low-weight to improve continuity without over-locking the sequence."
                )

            with gr.Accordion("Preview One Frame", open=False):
                preview_position = gr.Slider(1, 99, value=50, step=1, label="Preview position between endpoints (%)")
                preview_button = gr.Button("🔍 Generate One AI In-Between Frame", variant="secondary")
                preview_image = gr.Image(label="AI preview frame", type="filepath")
                preview_log = gr.Textbox(label="Preview log", lines=3)

            with gr.Row():
                generate_button = gr.Button("🚀 Generate PNG In-Between Sequence", variant="primary", size="lg")
                stop_button = gr.Button("⛔ Stop", variant="stop", size="lg")

            with gr.Row():
                preview_sheet = gr.Image(label="Generated frame sheet", type="filepath")
                all_zip = gr.File(label="All PNG frames (.zip)")
                middle_zip = gr.File(label="AI in-between frames only (.zip)")
            metadata_file = gr.File(label="Generation metadata (.json)")
            run_log = gr.Textbox(label="Run log", lines=10)

            preview_button.click(
                generate_one,
                inputs=[first, last, action_prompt, negative_prompt, style_notes, model_family, checkpoint, ipadapter, clip_vision,
                        resolution, preview_position, seed, steps, cfg, denoise_start, denoise_peak, endpoint_strength, low_vram, use_latent_guide],
                outputs=[preview_image, preview_log],
            )
            generate_button.click(
                generate_sequence,
                inputs=[first, last, action_prompt, negative_prompt, style_notes, model_family, checkpoint, ipadapter, clip_vision,
                        resolution, fps, duration, seed, seed_mode, steps, cfg, denoise_start, denoise_peak, endpoint_strength, temporal_strength,
                        previous_guide_strength, curve, first_lock, last_lock, low_vram, use_latent_guide, use_previous_frame],
                outputs=[preview_sheet, all_zip, middle_zip, metadata_file, run_log],
            )
            stop_button.click(stop_generation, outputs=run_log)

        with gr.Tab("Setup"):
            gr.Markdown(
                "## Connect to ComfyUI Desktop\n\n"
                "FrameForge does not need the ComfyUI Desktop installation folder. It talks to the local ComfyUI API.\n\n"
                "1. Start ComfyUI Desktop.\n"
                "2. Make sure the ComfyUI page is running.\n"
                "3. Leave the URL as `http://127.0.0.1:8188` unless you changed the port.\n"
                "4. Click **Save & Test Connection**.\n"
                "5. Click **Refresh model lists** after installing models."
            )
            comfy_url_box = gr.Textbox(value=comfy_url(), label="ComfyUI URL")
            with gr.Row():
                save_url_btn = gr.Button("Save & Test Connection", variant="primary")
                open_btn = gr.Button("Open ComfyUI")
                refresh_btn = gr.Button("Refresh model lists")
            status_box = gr.Textbox(label="Status", value=check_comfy()[1], lines=2)
            caps_box = gr.Textbox(label="FrameForge workflow capability", value=capabilities_text(), lines=2)
            save_url_btn.click(save_comfy_url, inputs=comfy_url_box, outputs=status_box)
            open_btn.click(open_comfy, outputs=status_box)
            refresh_btn.click(refresh_model_lists, inputs=model_family, outputs=[checkpoint, ipadapter, clip_vision]).then(
                lambda: capabilities_text(), outputs=caps_box
            )
            model_family.change(sync_family, inputs=model_family, outputs=[checkpoint, ipadapter, clip_vision])
            gr.Markdown(
                "### 8 GB profile\n"
                "Start with **SD 1.5 / 640×360 / 12 FPS / 2 s / 16 steps / endpoint denoise 0.36 / middle denoise 0.68**. "
                "Every middle frame is a separate image-generation job, so a long sequence can take significant time.\n\n"
                "### Method notes\n"
                "The workflow weights the first and last CLIP/IP-Adapter embeddings before combining them, then applies the combined embedding once. "
                "Denoise is scheduled low near both endpoints and higher in the middle, while a small previous-frame anchor improves frame-to-frame continuity. "
                "This is an image-by-image workflow; no video model is used."
            )

    return demo


if __name__ == "__main__":
    app().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)
