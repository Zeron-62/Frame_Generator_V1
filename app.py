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
try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

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
    "default_denoise": 0.55,
    "default_denoise_start": 0.24,
    "default_denoise_peak": 0.55,
    "default_endpoint_strength": 0.72,
    "default_temporal_strength": 0.12,
    "default_previous_latent_strength": 0.0,
    "default_curve": "Smoothstep",
    "default_generation_strategy": "Sequential Stable",
    "default_model_family": "SD 1.5",
    "default_anima_diffusion": "anima-base-v1.0.safetensors",
    "default_anima_text_encoder": "qwen_3_06b_base.safetensors",
    "default_anima_vae": "qwen_image_vae.safetensors",
    "default_anima_ipadapter": "ip_adapter.safetensors",
    "default_anima_ip_strength": 0.65,
    "default_anima_ip_cfg_scale": 2.0,
    "default_anima_ref_size": 512,
    "default_anima_ref_morph": 0.15,
    "default_anima_denoise_start": 0.22,
    "default_anima_denoise_peak": 0.48,
    "default_anima_lllite_strength": 0.35,
    "default_anima_lllite_target_bias": 0.18,
    "default_anima_canny_low": 0.17,
    "default_anima_canny_high": 0.45,
    "low_vram": True,
    "use_latent_guide": False,
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


def scan_anima_models() -> dict[str, list[str]]:
    return {
        "diffusion": _node_model_choices("UNETLoader", "unet_name"),
        "text_encoder": _node_model_choices("CLIPLoader", "clip_name"),
        "vae": _node_model_choices("VAELoader", "vae_name"),
        "ipadapter": _node_model_choices("AnimaIPAdapterLoader", "ip_adapter_name"),
    }


def scan_anima_control_models() -> list[str]:
    """Return Anima LLLite model-patch choices from the native ComfyUI loader."""
    choices = _node_model_choices("ModelPatchLoader", "name")
    if choices:
        return choices
    # Compatibility with the kohya sd-scripts node, which reads models/controlnet/.
    return _node_model_choices("AnimaLLLiteApply_sdscripts", "lllite_name")


def guess_anima_control_model() -> str:
    choices = scan_anima_control_models()
    preferred = (
        "anima-lllite-any-test-like-v2.safetensors",
        "anima-lllite-any-test-like-v2-beta-epoch-03.safetensors",
        "anima-lllite-any-test-like-1-step2000.safetensors",
    )
    for term in preferred:
        for item in choices:
            if term.lower() == str(item).lower() or term.lower() in str(item).lower():
                return str(item)
    return str(choices[0]) if choices else ""


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


def guess_anima_models() -> tuple[str, str, str, str]:
    models = scan_anima_models()

    def pick(items: list[str], terms: tuple[str, ...], fallback: str = "") -> str:
        low_items = [(str(x).lower(), str(x)) for x in items]
        for term in terms:
            for low, original in low_items:
                if term in low:
                    return original
        return fallback or (items[0] if items else "")

    return (
        pick(models["diffusion"], ("anima-base-v1.0", "anima-base", "anima"), CONFIG.get("default_anima_diffusion", "")),
        pick(models["text_encoder"], ("qwen_3_06b_base", "qwen_3", "qwen"), CONFIG.get("default_anima_text_encoder", "")),
        pick(models["vae"], ("qwen_image_vae", "qwen"), CONFIG.get("default_anima_vae", "")),
        pick(models["ipadapter"], ("ip_adapter.safetensors", "anima", "ip_adapter"), CONFIG.get("default_anima_ipadapter", "")),
    )



def resolve_anima_ipadapter(selected: str | None = None) -> str:
    """Resolve the Anima IP-Adapter filename robustly for ComfyUI Desktop.

    Prefer the UI selection, then the configured filename, then filenames exposed
    by the AnimaIPAdapterLoader. This prevents an empty Gradio dropdown from
    blocking generation when the model is installed but the node API does not
    expose its choices during startup.
    """
    candidates = []
    if selected:
        candidates.append(str(selected).strip())
    configured = str(CONFIG.get("default_anima_ipadapter", "")).strip()
    if configured:
        candidates.append(configured)
    # Known working checkpoint used in the project.
    candidates.append("ip_adapter-Character_Reference-10.safetensors")
    choices = scan_anima_models().get("ipadapter", [])
    candidates.extend(str(x) for x in choices if str(x).strip())
    seen = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        return c
    return ""

def refresh_all_model_lists(family: str = "SD 1.5"):
    sd = scan_models()
    sd_cp, sd_ip, sd_clip = guess_models(family if family in {"SD 1.5", "SDXL"} else "SD 1.5")
    anima = scan_anima_models()
    a_diff, a_text, a_vae, a_ip = guess_anima_models()
    control = scan_anima_control_models()
    a_control = guess_anima_control_model()
    is_anima = family == "Anima"
    return (
        gr.update(choices=sd["checkpoint"], value=sd_cp or None),
        gr.update(choices=sd["ipadapter"], value=sd_ip or None),
        gr.update(choices=sd["clip_vision"], value=sd_clip or None),
        gr.update(choices=anima["diffusion"], value=a_diff or None),
        gr.update(choices=anima["text_encoder"], value=a_text or None),
        gr.update(choices=anima["vae"], value=a_vae or None),
        gr.update(choices=(anima["ipadapter"] or [str(CONFIG.get("default_anima_ipadapter", "ip_adapter-Character_Reference-10.safetensors"))]), value=a_ip or str(CONFIG.get("default_anima_ipadapter", "ip_adapter-Character_Reference-10.safetensors"))),
        gr.update(choices=control, value=a_control or None),
        gr.update(visible=not is_anima),
        gr.update(visible=is_anima),
    )


def sync_family(family: str):
    return refresh_all_model_lists(family)


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


def make_rgb_fallback_guide(first_path: Path, last_path: Path, t: float, width: int, height: int) -> Path:
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


def center_out_plan(total_frames: int) -> list[tuple[int, int, int]]:
    """Return target frames in recursive temporal-subdivision order.

    Each tuple is (target_index, left_anchor_index, right_anchor_index). Starting with the
    full endpoint interval and repeatedly splitting the largest remaining interval gives
    every generated frame a nearby pair of already-known temporal anchors.
    """
    if total_frames < 3:
        return []
    pending: list[tuple[int, int]] = [(0, total_frames - 1)]
    plan: list[tuple[int, int, int]] = []
    while pending:
        pending.sort(key=lambda pair: (pair[1] - pair[0]), reverse=True)
        lo, hi = pending.pop(0)
        if hi - lo <= 1:
            continue
        mid = (lo + hi) // 2
        if mid <= lo:
            mid = lo + 1
        if mid >= hi:
            mid = hi - 1
        plan.append((mid, lo, hi))
        if mid - lo > 1:
            pending.append((lo, mid))
        if hi - mid > 1:
            pending.append((mid, hi))
    return plan


def sequential_plan(total_frames: int) -> list[tuple[int, int, int]]:
    """Simple forward plan used as a compatibility option."""
    return [(n, n - 1 if n > 1 else 0, total_frames - 1) for n in range(1, total_frames - 1)]


def bracket_reference_weights(
    target_index: int,
    total_frames: int,
    left_index: int,
    right_index: int,
    endpoint_strength: float,
    local_strength: float,
    curve: str,
) -> list[tuple[str, float]]:
    """Compute stable global endpoint + nearby temporal-anchor weights.

    The global pair preserves the endpoint identity/style across the sequence. The local pair
    receives a smaller budget, biased toward whichever known frame is temporally closer.
    Duplicate references are removed so an endpoint is not unintentionally double-weighted.
    """
    t = target_index / max(1, total_frames - 1)
    endpoint_u = time_curve(t, curve)
    first_w = float(endpoint_strength) * (1.0 - endpoint_u)
    last_w = float(endpoint_strength) * endpoint_u

    specs: list[tuple[str, float]] = [("__FIRST__", first_w), ("__LAST__", last_w)]
    if local_strength > 0:
        dl = max(1, target_index - left_index)
        dr = max(1, right_index - target_index)
        total = dl + dr
        lw = float(local_strength) * (dr / total)
        rw = float(local_strength) * (dl / total)
        specs.extend([(f"__FRAME_{left_index}__", lw), (f"__FRAME_{right_index}__", rw)])

    merged: dict[str, float] = {}
    for key, weight in specs:
        if weight > 0:
            merged[key] = merged.get(key, 0.0) + float(weight)
    return list(merged.items())


def latent_interpolation_supported() -> tuple[str | None, str | None]:
    if node_available("LatentInterpolate"):
        return "LatentInterpolate", "ratio"
    if node_available("LatentBlend"):
        return "LatentBlend", "blend_factor"
    return None, None



def build_modern_graph(
    checkpoint: str,
    ipadapter_file: str,
    clip_vision_file: str,
    reference_images: list[tuple[str, float]],
    left_image: str,
    right_image: str,
    local_ratio: float,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
) -> dict[str, dict[str, Any]]:
    """Build a modern IP-Adapter graph with multi-reference embeddings and temporal latent bracketing."""
    graph: dict[str, dict[str, Any]] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["1", 1]}},
        "6": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": ipadapter_file}},
        "7": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": clip_vision_file}},
    }

    next_id = 10
    embeddings: list[tuple[str, str]] = []
    for image_name, weight in reference_images:
        load_id = str(next_id)
        next_id += 1
        enc_id = str(next_id)
        next_id += 1
        graph[load_id] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
        graph[enc_id] = {
            "class_type": "IPAdapterEncoder",
            "inputs": {
                "ipadapter": ["6", 0],
                "image": [load_id, 0],
                "weight": float(weight),
                "clip_vision": ["7", 0],
            },
        }
        embeddings.append((enc_id, enc_id))

    if not embeddings:
        raise RuntimeError("No IP-Adapter reference images were provided.")

    pos_source = embeddings[0][0]
    neg_source = embeddings[0][1]
    neg_source_is_encoded = True
    for enc_id, neg_id in embeddings[1:]:
        combine_pos = str(next_id)
        next_id += 1
        combine_neg = str(next_id)
        next_id += 1
        graph[combine_pos] = {
            "class_type": "IPAdapterCombineEmbeds",
            "inputs": {"embed1": [pos_source, 0], "embed2": [enc_id, 0], "method": "norm average"},
        }
        graph[combine_neg] = {
            "class_type": "IPAdapterCombineEmbeds",
            "inputs": {
                "embed1": [neg_source, 1] if neg_source_is_encoded else [neg_source, 0],
                "embed2": [neg_id, 1],
                "method": "norm average",
            },
        }
        pos_source = combine_pos
        neg_source = combine_neg
        neg_source_is_encoded = False

    # Encode the two nearest known temporal anchors into the latent space. This replaces the old
    # RGB cross-fade guide whenever current ComfyUI exposes LatentInterpolate/LatentBlend.
    left_load = str(next_id)
    right_load = str(next_id + 1)
    left_vae = str(next_id + 2)
    right_vae = str(next_id + 3)
    next_id += 4
    graph[left_load] = {"class_type": "LoadImage", "inputs": {"image": left_image}}
    graph[right_load] = {"class_type": "LoadImage", "inputs": {"image": right_image}}
    graph[left_vae] = {"class_type": "VAEEncode", "inputs": {"pixels": [left_load, 0], "vae": ["1", 2]}}
    graph[right_vae] = {"class_type": "VAEEncode", "inputs": {"pixels": [right_load, 0], "vae": ["1", 2]}}

    latent_node, ratio_name = latent_interpolation_supported()
    if latent_node == "LatentInterpolate":
        latent_id = str(next_id)
        next_id += 1
        graph[latent_id] = {
            "class_type": "LatentInterpolate",
            "inputs": {
                "samples1": [left_vae, 0],
                "samples2": [right_vae, 0],
                "ratio": float(1.0 - local_ratio),
            },
        }
        latent_source = [latent_id, 0]
    elif latent_node == "LatentBlend":
        latent_id = str(next_id)
        next_id += 1
        graph[latent_id] = {
            "class_type": "LatentBlend",
            "inputs": {
                "samples1": [left_vae, 0],
                "samples2": [right_vae, 0],
                "blend_factor": float(local_ratio),
            },
        }
        latent_source = [latent_id, 0]
    else:
        raise RuntimeError(
            "This ComfyUI build does not expose LatentInterpolate or LatentBlend. "
            "Update ComfyUI, then retry the generation."
        )

    ipa_id = str(next_id)
    next_id += 1
    graph[ipa_id] = {
        "class_type": "IPAdapterEmbeds",
        "inputs": {
            "model": ["1", 0],
            "ipadapter": ["6", 0],
            "pos_embed": [pos_source, 0],
            "weight": 1.0,
            "weight_type": "linear",
            "start_at": 0.0,
            "end_at": 1.0,
            "embeds_scaling": "V only",
            "neg_embed": [neg_source, 1] if neg_source_is_encoded else [neg_source, 0],
            "clip_vision": ["7", 0],
        },
    }
    sampler_id = str(next_id)
    next_id += 1
    graph[sampler_id] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": float(denoise),
            "model": [ipa_id, 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": latent_source,
        },
    }
    decoded_id = str(next_id)
    next_id += 1
    graph[decoded_id] = {"class_type": "VAEDecode", "inputs": {"samples": [sampler_id, 0], "vae": ["1", 2]}}
    save_id = str(next_id)
    graph[save_id] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "FrameGeneratorV1", "images": [decoded_id, 0]}}
    return graph


def build_legacy_graph(
    checkpoint: str,
    ipadapter_file: str,
    clip_vision_file: str,
    reference_image: str,
    guide_image: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
    reference_weight: float,
) -> dict[str, dict[str, Any]]:
    """Compatibility graph for older IP-Adapter node packs."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": guide_image}},
        "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["1", 2]}},
        "6": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": ipadapter_file}},
        "7": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": clip_vision_file}},
        "8": {"class_type": "LoadImage", "inputs": {"image": reference_image}},
        "9": {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": ["1", 0], "ipadapter": ["6", 0], "clip_vision": ["7", 0],
                "image": ["8", 0], "weight": float(reference_weight), "weight_type": "linear",
                "combine_embeds": "average", "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only",
            },
        },
        "10": {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(seed), "steps": int(steps), "cfg": float(cfg), "sampler_name": "euler",
                "scheduler": "normal", "denoise": float(denoise), "model": ["9", 0],
                "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["5", 0],
            },
        },
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "FrameGeneratorV1", "images": ["11", 0]}},
    }


def build_stable_sequential_graph(
    checkpoint: str,
    ipadapter_file: str,
    clip_vision_file: str,
    first_image: str,
    last_image: str,
    previous_image: str | None,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
    first_weight: float,
    last_weight: float,
    previous_weight: float,
) -> dict[str, dict[str, Any]]:
    """Stability-first graph: previous-frame img2img anchor + two endpoint IP-Adapters.

    This deliberately avoids latent endpoint interpolation and multi-embedding normalization.
    Each frame starts from the immediately previous frame, while both supplied endpoints provide
    low-weight global identity/style constraints. This is easier to reason about and reduces the
    ghosting seen when several spatial guides are combined.
    """
    graph: dict[str, dict[str, Any]] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["1", 1]}},
        "4": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": ipadapter_file}},
        "5": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": clip_vision_file}},
        "6": {"class_type": "LoadImage", "inputs": {"image": previous_image or first_image}},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "LoadImage", "inputs": {"image": first_image}},
        "9": {"class_type": "LoadImage", "inputs": {"image": last_image}},
    }
    next_id = 10
    model_node = "1"

    # Apply first endpoint reference.
    first_node = str(next_id); next_id += 1
    graph[first_node] = {
        "class_type": "IPAdapterAdvanced",
        "inputs": {
            "model": [model_node, 0],
            "ipadapter": ["4", 0],
            "image": ["8", 0],
            "weight": float(max(0.0, min(1.0, first_weight))),
            "weight_type": "linear",
            "combine_embeds": "average",
            "start_at": 0.0,
            "end_at": 1.0,
            "embeds_scaling": "V only",
            "clip_vision": ["5", 0],
        },
    }
    model_node = first_node

    # Apply last endpoint reference.
    last_node = str(next_id); next_id += 1
    graph[last_node] = {
        "class_type": "IPAdapterAdvanced",
        "inputs": {
            "model": [model_node, 0],
            "ipadapter": ["4", 0],
            "image": ["9", 0],
            "weight": float(max(0.0, min(1.0, last_weight))),
            "weight_type": "linear",
            "combine_embeds": "average",
            "start_at": 0.0,
            "end_at": 1.0,
            "embeds_scaling": "V only",
            "clip_vision": ["5", 0],
        },
    }
    model_node = last_node

    # Optional previous-frame visual anchor. Kept intentionally low.
    if previous_image and previous_weight > 0.0:
        prev_load = str(next_id); next_id += 1
        prev_node = str(next_id); next_id += 1
        graph[prev_load] = {"class_type": "LoadImage", "inputs": {"image": previous_image}}
        graph[prev_node] = {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": [model_node, 0],
                "ipadapter": ["4", 0],
                "image": [prev_load, 0],
                "weight": float(max(0.0, min(0.35, previous_weight))),
                "weight_type": "linear",
                "combine_embeds": "average",
                "start_at": 0.0,
                "end_at": 1.0,
                "embeds_scaling": "V only",
                "clip_vision": ["5", 0],
            },
        }
        model_node = prev_node

    sampler = str(next_id); next_id += 1
    graph[sampler] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": float(max(0.05, min(0.95, denoise))),
            "model": [model_node, 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["7", 0],
        },
    }
    decoded = str(next_id); next_id += 1
    graph[decoded] = {"class_type": "VAEDecode", "inputs": {"samples": [sampler, 0], "vae": ["1", 2]}}
    save = str(next_id)
    graph[save] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "FrameGeneratorV1", "images": [decoded, 0]}}
    return graph


def api_graph_stable(
    checkpoint: str, ipadapter_file: str, clip_vision_file: str,
    first_image: str, last_image: str, previous_image: str | None,
    prompt: str, negative_prompt: str, seed: int, steps: int, cfg: float,
    denoise: float, first_weight: float, last_weight: float, previous_weight: float,
) -> dict[str, dict[str, Any]]:
    if not node_available("IPAdapterAdvanced"):
        raise RuntimeError("IPAdapterAdvanced is required for the stable workflow. Update ComfyUI_IPAdapter_plus.")
    return build_stable_sequential_graph(
        checkpoint, ipadapter_file, clip_vision_file, first_image, last_image, previous_image,
        prompt, negative_prompt, seed, steps, cfg, denoise, first_weight, last_weight, previous_weight
    )

def make_anima_reference(previous_path: Path, first_path: Path, last_path: Path, t: float, morph_strength: float, width: int, height: int) -> Path:
    """Create a temporary reference image for Anima IP-Adapter.

    The generated reference is never exported. It starts from the previous frame for temporal
    identity, then adds a small amount of first↔last endpoint information to indicate motion direction.
    """
    previous = prep_image(previous_path, width, height)
    first = prep_image(first_path, width, height)
    last = prep_image(last_path, width, height)
    endpoint_hint = Image.blend(first, last, float(max(0.0, min(1.0, t))))
    ref = Image.blend(previous, endpoint_hint, float(max(0.0, min(0.35, morph_strength))))
    path = TMP_DIR / f"anima_ref_{uuid.uuid4().hex[:10]}.png"
    ref.save(path, "PNG")
    return path


def anima_available() -> tuple[bool, list[str]]:
    required = ["UNETLoader", "CLIPLoader", "VAELoader", "AnimaIPAdapterLoader", "AnimaIPAdapterApply"]
    missing = [n for n in required if not node_available(n)]
    return (not missing, missing)


def anima_lllite_available() -> tuple[bool, list[str], str]:
    """Return availability for the v8.4 hybrid Anima path.

    v8.4 intentionally avoids using the Anima LLLite control patch as the primary
    structural mechanism. Instead, it uses a dense optical-flow guide as BOTH:
      1) an image-to-image latent initialization, and
      2) the Anima IP-Adapter reference image.
    This keeps semantic image content in the generation path and avoids the
    sparse-edge collapse observed with Canny-only LLLite control.
    """
    required = ["UNETLoader", "CLIPLoader", "VAELoader",
                "VAEEncode", "AnimaIPAdapterLoader", "AnimaIPAdapterApply"]
    missing = [n for n in required if not node_available(n)]
    return (not missing, missing, "hybrid" if not missing else "")


def make_anima_control_guide(previous_path: Path, last_path: Path, t: float, width: int, height: int, target_bias: float) -> Path:
    """Create a non-recursive structural guide that actually moves toward the last frame.

    v8.3 uses dense optical flow from the first endpoint toward the last endpoint. A frame at
    time t is produced by warping the first frame along t * flow, giving LLLite a concrete
    destination trajectory without alpha-blending two poses (which can create double edges).

    The previous frame is retained as a fallback only when optical-flow construction fails.
    The guide is temporary and never exported as a final frame.
    """
    first = prep_image(previous_path, width, height)
    target = prep_image(last_path, width, height)
    if cv2 is not None and np is not None:
        try:
            src = np.array(first, dtype=np.uint8)
            dst = np.array(target, dtype=np.uint8)
            g1 = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
            g2 = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                g1, g2, None, 0.5, 3, 21, 3, 5, 1.2, 0
            )
            h, w = g1.shape
            grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            # Conservative trajectory; avoid large extrapolations outside the image.
            scale_t = float(max(0.0, min(1.0, t)))
            map_x = np.clip(grid_x + flow[..., 0] * scale_t, 0, w - 1).astype(np.float32)
            map_y = np.clip(grid_y + flow[..., 1] * scale_t, 0, h - 1).astype(np.float32)
            warped = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            # Small endpoint pull near the destination reduces flow approximation error without
            # reintroducing a full cross-fade or doubled geometry.
            endpoint_pull = min(0.20, max(0.0, scale_t - 0.70) * 0.65)
            if endpoint_pull > 0:
                warped = cv2.addWeighted(warped, 1.0 - endpoint_pull, dst, endpoint_pull, 0.0)
            guide = Image.fromarray(warped)
        except Exception:
            guide = first
    else:
        guide = first
    path = TMP_DIR / f"anima_control_{uuid.uuid4().hex[:10]}.png"
    guide.save(path, "PNG")
    return path


def build_anima_lllite_graph(
    diffusion_model: str,
    text_encoder: str,
    vae_name: str,
    control_model: str,
    control_image: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    control_strength: float,
    canny_low: float,
    canny_high: float,
    width: int,
    height: int,
    denoise: float = 1.0,
    turbo_mode: bool = False,
    turbo_lora: str | None = None,
    ipadapter_name: str | None = None,
    ip_strength: float = 0.45,
    ip_cfg_scale: float = 1.6,
    ref_size: int = 512,
) -> dict[str, dict[str, Any]]:
    """Build the v8.4 hybrid Anima graph.

    The structural guide is kept as a full RGB image rather than reduced to
    Canny edges. It is:
      - VAE-encoded to initialize the latent,
      - fed directly to Anima IP-Adapter as the reference image.

    This makes the model see the actual intermediate scene content instead of
    only sparse edges. The control_model / canny arguments remain in the
    function signature for compatibility with the v8.x UI, but are unused.
    """
    ok, missing = anima_available()
    if not ok:
        raise RuntimeError(
            "Anima support is not available. Missing ComfyUI nodes: " + ", ".join(missing)
        )
    ipadapter_name = resolve_anima_ipadapter(ipadapter_name)
    if not ipadapter_name:
        raise RuntimeError("No Anima IP-Adapter model is available. Refresh model lists or place the checkpoint in the ComfyUI ipadapter folder.")

    graph: dict[str, dict[str, Any]] = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": diffusion_model, "weight_dtype": "default"
        }},
        "2": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": text_encoder, "type": "stable_diffusion"
        }},
        "3": {"class_type": "VAELoader", "inputs": {
            "vae_name": vae_name
        }},
        "4": {"class_type": "LoadImage", "inputs": {
            "image": control_image
        }},
        "5": {"class_type": "VAEEncode", "inputs": {
            "pixels": ["4", 0],
            "vae": ["3", 0]
        }},
        "6": {"class_type": "CLIPTextEncode", "inputs": {
            "text": prompt, "clip": ["2", 0]
        }},
        "7": {"class_type": "CLIPTextEncode", "inputs": {
            "text": negative_prompt, "clip": ["2", 0]
        }},
        "8": {"class_type": "AnimaIPAdapterLoader", "inputs": {
            "ip_adapter_name": ipadapter_name,
            "auto_download": False
        }},
        "9": {"class_type": "AnimaIPAdapterApply", "inputs": {
            "model": ["1", 0],
            "ip_adapter": ["8", 0],
            "ref_image": ["4", 0],
            "strength": float(ip_strength),
            "ref_image_size": int(ref_size),
            "siglip_layer": -1,
            "ip_cfg_scale": float(max(1.01, ip_cfg_scale)),
            "ip_cfg_separate": False,
            "gray_null": False,
            "use_lora": bool(turbo_mode),
        }},
        "10": {"class_type": "KSampler", "inputs": {
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": float(max(0.10, min(0.75, denoise))),
            "model": ["9", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        }},
        "11": {"class_type": "VAEDecode", "inputs": {
            "samples": ["10", 0],
            "vae": ["3", 0]
        }},
        "12": {"class_type": "SaveImage", "inputs": {
            "filename_prefix": "FrameGeneratorV1_AnimaHybrid",
            "images": ["11", 0]
        }},
    }

    return graph

def generation_prompt_anima_v8(action_prompt: str, t: float, style_notes: str) -> str:
    extra = style_notes.strip()
    return (
        f"{action_prompt.strip()}. This is intermediate animation frame {t*100:.1f}% between the endpoints. "
        "Preserve the subject identity, face, hair silhouette, clothing, body proportions, camera, framing, "
        "background palette and lighting continuity. Use the provided structure control as geometry guidance. "
        "Change only what is required by the motion. Keep anatomy coherent and transitions gradual; no teleporting, "
        "double exposure, duplicate limbs, costume changes, camera jumps or new characters."
        + (f" {extra}." if extra else "")
    )


def generate_one_anima_lllite(
    first: str | None, last: str | None, action_prompt: str, negative_prompt: str, style_notes: str,
    diffusion_model: str, text_encoder: str, vae_name: str, control_model: str, ipadapter_name: str | None,
    resolution: str, position: float, seed: float, steps: int, cfg: float,
    denoise_start: float, denoise_peak: float, control_strength: float, target_bias: float,
    canny_low: float, canny_high: float, progress=gr.Progress(track_tqdm=False),
):
    STOP_EVENT.clear()
    try:
        if not first or not last:
            raise gr.Error("Upload both a first frame and a last frame.")
        if not action_prompt.strip():
            raise gr.Error("Enter an action / motion prompt.")
        ipadapter_name = resolve_anima_ipadapter(ipadapter_name)
        ok, missing, mode = anima_lllite_available()
        if not ok:
            raise gr.Error("Anima hybrid mode is unavailable: " + ", ".join(missing))
        # v8.4 hybrid mode keeps the legacy control_model field for UI compatibility,
        # but it is not required for generation.
        width, height = (int(x) for x in resolution.lower().split("x"))
        first_path = normalize_input(first, "first_frame")
        last_path = normalize_input(last, "last_frame")
        t = max(0.001, min(0.999, float(position) / 100.0))
        base_seed = secrets.randbelow(2**31 - 1) if int(seed) < 0 else int(seed)
        denoise = adaptive_denoise(t, float(denoise_start), float(denoise_peak), "Smoothstep")
        guide_path = make_anima_control_guide(first_path, last_path, t, width, height, target_bias)
        ref_upload = upload_image(first_path)
        guide_upload = upload_image(guide_path)
        graph = build_anima_lllite_graph(
            diffusion_model, text_encoder, vae_name, control_model, guide_upload,
            generation_prompt_anima_v8(action_prompt, t, style_notes), negative_prompt.strip(),
            int(base_seed), int(steps), min(5.0, float(cfg)), float(control_strength),
            float(canny_low), float(canny_high), width, height, denoise=denoise,
            ipadapter_name=ipadapter_name,
            ip_strength=float(CONFIG.get("default_anima_ip_strength", 0.45)),
            ip_cfg_scale=float(CONFIG.get("default_anima_ip_cfg_scale", 1.6)),
            ref_size=int(CONFIG.get("default_anima_ref_size", 512)),
        )
        progress(0.10, desc="Submitting Anima hybrid preview…")
        pid = submit(graph)
        frame = wait_for_image(pid, lambda: progress(0.30, desc="Generating Anima hybrid preview…"))
        out_path = OUTPUT_DIR / f"FrameGeneratorV1_anima_lllite_preview_{uuid.uuid4().hex[:10]}.png"
        shutil.copy2(frame, out_path)
        return str(out_path), (
            f"Anima dual-endpoint preview: {position:.1f}%\nSeed: {base_seed}\nResolution: {width}x{height}\n"
            f"Denoise: {denoise:.3f}\nControl strength: {control_strength:.2f}\nTarget bias: {target_bias:.2f}\n"
            f"Canny: {canny_low:.2f} → {canny_high:.2f}\nBackend: {mode}\n"
        )
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def generate_sequence_anima_lllite(
    first: str | None, last: str | None, action_prompt: str, negative_prompt: str, style_notes: str,
    diffusion_model: str, text_encoder: str, vae_name: str, control_model: str, ipadapter_name: str | None, resolution: str,
    fps: int, duration: float, seed: float, seed_mode: str, steps: int, cfg: float,
    denoise_start: float, denoise_peak: float, control_strength: float, target_bias: float,
    canny_low: float, canny_high: float, progress=gr.Progress(track_tqdm=False),
):
    STOP_EVENT.clear()
    try:
        if not first or not last:
            raise gr.Error("Upload both a first frame and a last frame.")
        if not action_prompt.strip():
            raise gr.Error("Enter an action / motion prompt.")
        ipadapter_name = resolve_anima_ipadapter(ipadapter_name)
        ok, missing, mode = anima_lllite_available()
        if not ok:
            raise gr.Error("Anima hybrid mode is unavailable: " + ", ".join(missing))
        # v8.4 hybrid mode keeps the legacy control_model field for UI compatibility,
        # but it is not required for generation.
        width, height = (int(x) for x in resolution.lower().split("x"))
        total_frames = max(3, int(round(float(fps) * float(duration))))
        if total_frames > 120:
            raise gr.Error("This build caps a single job at 120 total frames.")
        if float(denoise_peak) < float(denoise_start):
            raise gr.Error("Middle denoise must be greater than or equal to endpoint denoise.")
        base_seed = secrets.randbelow(2**31 - 1) if int(seed) < 0 else int(seed)
        first_path = normalize_input(first, "first_frame")
        last_path = normalize_input(last, "last_frame")
        job_id = uuid.uuid4().hex[:10]
        render_dir = FRAMES_DIR / job_id
        clean_sequence(render_dir)
        render_dir.mkdir(parents=True, exist_ok=True)
        first_out = render_dir / "frame_0001.png"
        last_out = render_dir / f"frame_{total_frames:04d}.png"
        shutil.copy2(first_path, first_out)
        shutil.copy2(last_path, last_out)
        anchors: dict[int, Path] = {0: first_path}
        generated: list[Path] = []
        uploaded: dict[str, str] = {}
        last_known_good = first_path
        def cached_upload(path: Path) -> str:
            key = str(path.resolve())
            if key not in uploaded:
                uploaded[key] = upload_image(path)
            return uploaded[key]
        def frame_is_usable(path: Path) -> bool:
            """Reject genuine black/corrupt outputs without requiring NumPy."""
            try:
                im = Image.open(path).convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
                pixels = list(im.getdata())
                if not pixels:
                    return False
                mean = sum((r + g + b) / 3.0 for r, g, b in pixels) / len(pixels)
                near_black = sum(1 for r, g, b in pixels if max(r, g, b) < 3) / len(pixels)
                # Reject genuine black collapse while allowing very dark night scenes.
                return mean > 4.0 and near_black < 0.96
            except Exception:
                return False
        step_count = total_frames - 2
        progress(0.03, desc=f"Generating {step_count} Anima hybrid in-between frames…")
        for step_no, target_index in enumerate(range(1, total_frames - 1), start=1):
            if STOP_EVENT.is_set():
                interrupt_comfy()
                raise RuntimeError("Generation stopped by user.")
            previous = last_known_good
            t = target_index / (total_frames - 1)
            guide_path = make_anima_control_guide(first_path, last_path, t, width, height, target_bias)
            guide_upload = cached_upload(guide_path)
            frame_seed = base_seed if seed_mode == "Fixed" else (base_seed + target_index * 7919) % (2**31 - 1)
            current_index = target_index + 1

            # First attempt: conservative official-style LLLite control-to-image.
            graph = build_anima_lllite_graph(
                diffusion_model, text_encoder, vae_name, control_model, guide_upload,
                generation_prompt_anima_v8(action_prompt, t, style_notes), negative_prompt.strip(),
                int(frame_seed), int(steps), min(5.0, float(cfg)), float(control_strength),
                float(canny_low), float(canny_high), width, height, denoise=adaptive_denoise(t, float(denoise_start), float(denoise_peak), "Smoothstep"),
                ipadapter_name=ipadapter_name,
            )
            pid = submit(graph)
            tmp = wait_for_image(pid, lambda: progress(
                0.05 + 0.90 * ((step_no - 1) / max(1, step_count)),
                desc=f"Anima hybrid frame {current_index}/{total_frames} — {t*100:.1f}%"
            ))

            # Automatic black-frame recovery: retry from the last known-good control frame
            # with reduced LLLite strength and slightly stronger sampling before accepting.
            if not frame_is_usable(tmp):
                recovery_strength = max(0.15, min(float(control_strength), float(control_strength) * 0.55))
                recovery_steps = max(18, int(steps) + 4)
                recovery_seed = (int(frame_seed) + 104729) % (2**31 - 1)
                recovery_graph = build_anima_lllite_graph(
                    diffusion_model, text_encoder, vae_name, control_model, guide_upload,
                    generation_prompt_anima_v8(action_prompt, t, style_notes), negative_prompt.strip(),
                    recovery_seed, recovery_steps, min(5.0, float(cfg)), recovery_strength,
                    float(canny_low), float(canny_high), width, height,
                    denoise=adaptive_denoise(t, float(denoise_start), float(denoise_peak), "Smoothstep"),
                    ipadapter_name=ipadapter_name,
                    ip_strength=float(CONFIG.get("default_anima_ip_strength", 0.45)),
                    ip_cfg_scale=float(CONFIG.get("default_anima_ip_cfg_scale", 1.6)),
                    ref_size=int(CONFIG.get("default_anima_ref_size", 512)),
                )
                recovery_pid = submit(recovery_graph)
                tmp = wait_for_image(recovery_pid, lambda: progress(
                    min(0.98, 0.05 + 0.90 * ((step_no - 0.5) / max(1, step_count))),
                    desc=f"Recovering Anima hybrid frame {current_index}/{total_frames}…"
                ))

            dest = render_dir / f"frame_{current_index:04d}.png"
            shutil.copy2(tmp, dest)
            if frame_is_usable(dest):
                anchors[target_index] = dest
                last_known_good = dest
                generated.append(dest)
            else:
                # Never feed a collapsed frame into the next step. Keep the endpoint-safe
                # previous frame and record a duplicate only as a last-resort safety fallback.
                safe_dest = render_dir / f"frame_{current_index:04d}.png"
                shutil.copy2(last_known_good, safe_dest)
                anchors[target_index] = last_known_good
                generated.append(safe_dest)
        all_frames = sorted([first_out] + generated + [last_out], key=lambda p: int(p.stem.split('_')[-1]))
        all_zip = OUTPUT_DIR / f"FrameGeneratorV1_{job_id}_all_frames.zip"
        middle_zip = OUTPUT_DIR / f"FrameGeneratorV1_{job_id}_inbetween_only.zip"
        contact = OUTPUT_DIR / f"FrameGeneratorV1_{job_id}_contact_sheet.jpg"
        metadata = OUTPUT_DIR / f"FrameGeneratorV1_{job_id}_metadata.json"
        zip_frames(all_frames, all_zip)
        zip_frames(generated, middle_zip)
        save_contact_sheet(all_frames, contact)
        metadata.write_text(json.dumps({
            "job_id": job_id, "project": "Frame_Generator_V1", "version": "v8.4.1",
            "model_family": "Anima", "anima_engine": "Anima hybrid flow/img2img + IP-Adapter",
            "diffusion_model": diffusion_model, "text_encoder": text_encoder, "vae": vae_name,
            "control_model": control_model, "total_frames": total_frames, "ai_inbetween_frames": step_count,
            "fps": fps, "duration_seconds": float(duration), "resolution": [width, height],
            "seed": base_seed, "seed_mode": seed_mode, "steps": steps, "cfg": float(min(5.0, cfg)),
            "denoise_start": float(denoise_start), "denoise_peak": float(denoise_peak),
            "control_strength": float(control_strength), "target_bias": float(target_bias),
            "canny_low": float(canny_low), "canny_high": float(canny_high),
            "action_prompt": action_prompt, "negative_prompt": negative_prompt, "consistency_notes": style_notes,
            "method": "Previous-frame VAE img2img anchor + time-weighted first/last endpoint Anima IP-Adapter conditioning; PNG frames only"
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        log_text = (
            f"Job: {job_id}\nModel: Anima Base v1.0 + sequential img2img + dual endpoint IP-Adapter\nTotal frames: {total_frames}\n"
            f"AI-generated middle frames: {step_count}\nFPS / duration: {fps} / {duration:.2f}s\n"
            f"Resolution: {width}x{height}\nSeed: {base_seed} ({seed_mode})\n"
            f"Denoise: {denoise_start:.2f} → {denoise_peak:.2f}\nControl strength: {control_strength:.2f}\n"
            f"Target bias: {target_bias:.2f}\nCanny: {canny_low:.2f} → {canny_high:.2f}\n"
            "No MP4/video export is generated.\n"
            f"Output folder: {render_dir}\n"
        )
        progress(1.0, desc="Done — Anima LLLite PNG frame sequence created")
        return str(contact), str(all_zip), str(middle_zip), str(metadata), log_text
    except gr.Error:
        raise
    except Exception as exc:
        progress(1.0, desc="Generation stopped or failed")
        raise gr.Error(str(exc)) from exc


def build_anima_graph(
    diffusion_model: str,
    text_encoder: str,
    vae_name: str,
    ipadapter_name: str,
    first_reference: str,
    last_reference: str,
    latent_reference: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
    ip_strength: float,
    ip_cfg_scale: float,
    ref_size: int,
    t: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Build a sequential Anima img2img graph with separate endpoint references.

    Design:
      - latent_reference = previous frame (local temporal anchor)
      - first_reference = original first endpoint
      - last_reference = original last endpoint
      - two Anima IP-Adapter applications provide time-varying endpoint guidance
        without pixel cross-fading the reference images.
    """
    ok, missing = anima_available()
    if not ok:
        raise RuntimeError(
            "Anima support is not available. Missing ComfyUI nodes: " + ", ".join(missing) +
            ". Install the Anima IP-Adapter node and restart ComfyUI."
        )
    ipadapter_name = resolve_anima_ipadapter(ipadapter_name)
    if not ipadapter_name:
        raise RuntimeError("No Anima IP-Adapter model is available. Refresh model lists or place the checkpoint in the ComfyUI ipadapter folder.")

    t = max(0.0, min(1.0, float(t)))
    total_endpoint_strength = max(0.05, float(ip_strength))
    first_strength = total_endpoint_strength * (1.0 - t)
    last_strength = total_endpoint_strength * t

    graph: dict[str, dict[str, Any]] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": diffusion_model, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": text_encoder, "type": "stable_diffusion"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["2", 0]}},
        "6": {"class_type": "LoadImage", "inputs": {"image": latent_reference}},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "LoadImage", "inputs": {"image": first_reference}},
        "9": {"class_type": "LoadImage", "inputs": {"image": last_reference}},
        "10": {"class_type": "AnimaIPAdapterLoader", "inputs": {"ip_adapter_name": ipadapter_name, "auto_download": False}},
    }

    current_model_node = "1"
    next_id = 11

    if first_strength > 0.01:
        graph[str(next_id)] = {"class_type": "AnimaIPAdapterApply", "inputs": {
            "model": [current_model_node, 0],
            "ip_adapter": ["10", 0],
            "ref_image": ["8", 0],
            "strength": float(first_strength),
            "ref_image_size": int(ref_size),
            "siglip_layer": -1,
            "ip_cfg_scale": float(max(1.01, ip_cfg_scale)),
            "ip_cfg_separate": False,
            "gray_null": False,
            "use_lora": True,
        }}
        current_model_node = str(next_id)
        next_id += 1

    if last_strength > 0.01:
        graph[str(next_id)] = {"class_type": "AnimaIPAdapterApply", "inputs": {
            "model": [current_model_node, 0],
            "ip_adapter": ["10", 0],
            "ref_image": ["9", 0],
            "strength": float(last_strength),
            "ref_image_size": int(ref_size),
            "siglip_layer": -1,
            "ip_cfg_scale": float(max(1.01, ip_cfg_scale)),
            "ip_cfg_separate": False,
            "gray_null": False,
            "use_lora": True,
        }}
        current_model_node = str(next_id)
        next_id += 1

    sampler_id = next_id
    vaedecode_id = next_id + 1
    save_id = next_id + 2
    graph[str(sampler_id)] = {"class_type": "KSampler", "inputs": {
        "seed": int(seed),
        "steps": int(steps),
        "cfg": float(cfg),
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": float(max(0.05, min(0.85, denoise))),
        "model": [current_model_node, 0],
        "positive": ["4", 0],
        "negative": ["5", 0],
        "latent_image": ["7", 0],
    }}
    graph[str(vaedecode_id)] = {"class_type": "VAEDecode", "inputs": {
        "samples": [str(sampler_id), 0],
        "vae": ["3", 0]
    }}
    graph[str(save_id)] = {"class_type": "SaveImage", "inputs": {
        "filename_prefix": "FrameGeneratorV1_Anima_DualEndpoint",
        "images": [str(vaedecode_id), 0]
    }}
    return graph


def generation_prompt_anima(action_prompt: str, t: float, style_notes: str) -> str:
    extra = style_notes.strip()
    return (
        f"{action_prompt.strip()}. This is an intermediate animation frame at {t*100:.1f}% of the action. "
        "Preserve the same character identity, face, eyes, hair silhouette, outfit, body proportions, camera, framing, "
        "background and lighting. Change only what is required by the action. Smooth pose progression, coherent anatomy, "
        "stable anime line art and color palette, no sudden pose jump, no duplicate body parts, no new character."
        + (f" {extra}." if extra else "")
    )


def generate_one_anima(
    first: str | None,
    last: str | None,
    action_prompt: str,
    negative_prompt: str,
    style_notes: str,
    diffusion_model: str,
    text_encoder: str,
    vae_name: str,
    ipadapter_name: str,
    resolution: str,
    position: float,
    seed: float,
    steps: int,
    cfg: float,
    denoise_start: float,
    denoise_peak: float,
    ip_strength: float,
    ip_cfg_scale: float,
    ref_size: int,
    ref_morph: float,
    progress=gr.Progress(track_tqdm=False),
):
    STOP_EVENT.clear()
    try:
        if not first or not last:
            raise gr.Error("Upload both a first frame and a last frame.")
        if not action_prompt.strip():
            raise gr.Error("Enter an action / motion prompt.")
        ok, missing = anima_available()
        if not ok:
            raise gr.Error("Missing Anima nodes: " + ", ".join(missing))
        width, height = (int(x) for x in resolution.lower().split("x"))
        first_path = normalize_input(first, "first_frame")
        last_path = normalize_input(last, "last_frame")
        t = max(0.001, min(0.999, float(position) / 100.0))
        base_seed = secrets.randbelow(2**31 - 1) if int(seed) < 0 else int(seed)
        denoise = adaptive_denoise(t, float(denoise_start), float(denoise_peak), "Smootherstep")
        first_upload = upload_image(first_path)
        last_upload = upload_image(last_path)
        graph = build_anima_graph(
            diffusion_model, text_encoder, vae_name, first_upload, last_upload, first_upload,
            generation_prompt_anima(action_prompt, t, style_notes), negative_prompt.strip(),
            int(base_seed), int(steps), float(cfg), denoise, float(ip_strength), float(ip_cfg_scale), int(ref_size), t=t
        )
        progress(0.10, desc="Submitting Anima preview…")
        pid = submit(graph)
        frame = wait_for_image(pid, lambda: progress(0.30, desc="Generating Anima preview…"))
        out_path = OUTPUT_DIR / f"FrameGeneratorV1_anima_preview_{uuid.uuid4().hex[:10]}.png"
        shutil.copy2(frame, out_path)
        return str(out_path), (
            f"Anima preview: {position:.1f}%\nSeed: {base_seed}\nResolution: {width}x{height}\n"
            f"Denoise: {denoise:.3f}\nIP strength: {ip_strength:.2f}\nIP CFG: {ip_cfg_scale:.2f}\n"
            f"Reference morph: {ref_morph:.2f}\n"
        )
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def generate_sequence_anima(
    first: str | None,
    last: str | None,
    action_prompt: str,
    negative_prompt: str,
    style_notes: str,
    diffusion_model: str,
    text_encoder: str,
    vae_name: str,
    ipadapter_name: str,
    resolution: str,
    fps: int,
    duration: float,
    seed: float,
    seed_mode: str,
    steps: int,
    cfg: float,
    denoise_start: float,
    denoise_peak: float,
    ip_strength: float,
    ip_cfg_scale: float,
    ref_size: int,
    ref_morph: float,
    progress=gr.Progress(track_tqdm=False),
):
    STOP_EVENT.clear()
    try:
        if not first or not last:
            raise gr.Error("Upload both a first frame and a last frame.")
        if not action_prompt.strip():
            raise gr.Error("Enter an action / motion prompt.")
        ok, missing = anima_available()
        if not ok:
            raise gr.Error("Missing Anima nodes: " + ", ".join(missing))
        width, height = (int(x) for x in resolution.lower().split("x"))
        if float(denoise_peak) < float(denoise_start):
            raise gr.Error("Middle denoise must be greater than or equal to endpoint denoise.")
        total_frames = max(3, int(round(float(fps) * float(duration))))
        if total_frames > 120:
            raise gr.Error("This build caps a single job at 120 total frames.")
        base_seed = secrets.randbelow(2**31 - 1) if int(seed) < 0 else int(seed)
        first_path = normalize_input(first, "first_frame")
        last_path = normalize_input(last, "last_frame")
        job_id = uuid.uuid4().hex[:10]
        render_dir = FRAMES_DIR / job_id
        clean_sequence(render_dir)
        render_dir.mkdir(parents=True, exist_ok=True)
        first_out = render_dir / "frame_0001.png"
        last_out = render_dir / f"frame_{total_frames:04d}.png"
        shutil.copy2(first_path, first_out)
        shutil.copy2(last_path, last_out)
        anchors: dict[int, Path] = {0: first_path, total_frames - 1: last_path}
        generated: list[Path] = []
        uploaded: dict[str, str] = {}

        def cached_upload(path: Path) -> str:
            key = str(path.resolve())
            if key not in uploaded:
                uploaded[key] = upload_image(path)
            return uploaded[key]

        step_count = total_frames - 2
        progress(0.03, desc=f"Generating {step_count} Anima in-between frames…")

        for step_no, target_index in enumerate(range(1, total_frames - 1), start=1):
            if STOP_EVENT.is_set():
                interrupt_comfy()
                raise RuntimeError("Generation stopped by user.")
            previous = anchors[target_index - 1]
            t = target_index / (total_frames - 1)
            denoise = adaptive_denoise(t, float(denoise_start), float(denoise_peak), "Smootherstep")
            prev_upload = cached_upload(previous)
            first_upload = cached_upload(first_path)
            last_upload = cached_upload(last_path)
            if seed_mode == "Fixed":
                frame_seed = base_seed
            else:
                frame_seed = (base_seed + target_index * 7919) % (2**31 - 1)
            graph = build_anima_graph(
                diffusion_model, text_encoder, vae_name, ipadapter_name,
                first_upload, last_upload, prev_upload,
                generation_prompt_anima(action_prompt, t, style_notes), negative_prompt.strip(),
                int(frame_seed), int(steps), float(cfg), denoise, float(ip_strength), float(ip_cfg_scale), int(ref_size), t=t
            )
            pid = submit(graph)
            current_index = target_index + 1
            tmp = wait_for_image(pid, lambda: progress(
                0.05 + 0.90 * ((step_no - 1) / max(1, step_count)),
                desc=f"Anima frame {current_index}/{total_frames} — {t*100:.1f}%"
            ))
            dest = render_dir / f"frame_{current_index:04d}.png"
            shutil.copy2(tmp, dest)
            anchors[target_index] = dest
            generated.append(dest)

        all_frames = [first_out] + sorted(generated, key=lambda p: int(p.stem.split('_')[-1])) + [last_out]
        all_frames = sorted(all_frames, key=lambda p: int(p.stem.split('_')[-1]))
        all_zip = OUTPUT_DIR / f"FrameGeneratorV1_{job_id}_all_frames.zip"
        middle_zip = OUTPUT_DIR / f"FrameGeneratorV1_{job_id}_inbetween_only.zip"
        contact = OUTPUT_DIR / f"FrameGeneratorV1_{job_id}_contact_sheet.jpg"
        metadata = OUTPUT_DIR / f"FrameGeneratorV1_{job_id}_metadata.json"
        zip_frames(all_frames, all_zip)
        zip_frames(generated, middle_zip)
        save_contact_sheet(all_frames, contact)
        metadata.write_text(json.dumps({
            "job_id": job_id, "project": "Frame_Generator_V1", "model_family": "Anima",
            "diffusion_model": diffusion_model, "text_encoder": text_encoder, "vae": vae_name,
            "anima_ipadapter": ipadapter_name, "total_frames": total_frames,
            "ai_inbetween_frames": step_count, "fps": fps, "duration_seconds": float(duration),
            "resolution": [width, height], "seed": base_seed, "seed_mode": seed_mode,
            "steps": steps, "cfg": cfg, "denoise_start": float(denoise_start),
            "denoise_peak": float(denoise_peak), "ip_strength": float(ip_strength),
            "ip_cfg_scale": float(ip_cfg_scale), "reference_size": int(ref_size),
            "reference_morph": float(ref_morph), "action_prompt": action_prompt,
            "negative_prompt": negative_prompt, "consistency_notes": style_notes,
            "method": "Anima Base v1.0 img2img + Anima SigLIP2 IP-Adapter + previous-frame img2img anchor + endpoint-morph reference guide; PNG frames only"
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        log_text = (
            f"Job: {job_id}\nModel: Anima Base v1.0\nTotal frames: {total_frames}\n"
            f"AI-generated middle frames: {step_count}\nFPS / duration: {fps} / {duration:.2f}s\n"
            f"Resolution: {width}x{height}\nSeed: {base_seed} ({seed_mode})\n"
            f"Denoise: {denoise_start:.2f} → {denoise_peak:.2f}\n"
            f"Anima IP strength: {ip_strength:.2f}\nAnima IP CFG: {ip_cfg_scale:.2f}\n"
            f"Reference morph: {ref_morph:.2f}\nNo MP4/video export is generated.\nOutput folder: {render_dir}\n"
        )
        progress(1.0, desc="Done — Anima PNG frame sequence created")
        return str(contact), str(all_zip), str(middle_zip), str(metadata), log_text
    except gr.Error:
        raise
    except Exception as exc:
        progress(1.0, desc="Generation stopped or failed")
        raise gr.Error(str(exc)) from exc


def generate_preview_dispatch(
    first, last, action_prompt, negative_prompt, style_notes, model_family,
    checkpoint, ipadapter, clip_vision,
    anima_diffusion, anima_text_encoder, anima_vae, anima_ipadapter, anima_control_model,
    resolution, position, seed, steps, cfg, denoise_start, denoise_peak,
    endpoint_strength, local_anchor_strength, anima_ip_strength, anima_ip_cfg_scale,
    anima_ref_size, anima_ref_morph, anima_lllite_strength, anima_target_bias, anima_canny_low, anima_canny_high,
):
    if model_family == "Anima":
        return generate_one_anima_lllite(
            first, last, action_prompt, negative_prompt, style_notes,
            anima_diffusion, anima_text_encoder, anima_vae, anima_control_model, anima_ipadapter,
            resolution, position, seed, steps, min(5.0, float(cfg)), denoise_start, denoise_peak,
            anima_lllite_strength, anima_target_bias, anima_canny_low, anima_canny_high,
        )
    if model_family == "Anima-IP":
        return generate_one_anima(
            first, last, action_prompt, negative_prompt, style_notes,
            anima_diffusion, anima_text_encoder, anima_vae, anima_ipadapter,
            resolution, position, seed, steps, min(5.0, float(cfg)), denoise_start, denoise_peak,
            anima_ip_strength, anima_ip_cfg_scale, anima_ref_size, anima_ref_morph,
        )
    return generate_one(
        first, last, action_prompt, negative_prompt, style_notes, model_family,
        checkpoint, ipadapter, clip_vision, resolution, position, seed, steps, cfg,
        denoise_start, denoise_peak, endpoint_strength, local_anchor_strength, True,
    )


def generate_sequence_dispatch(
    first, last, action_prompt, negative_prompt, style_notes, model_family,
    checkpoint, ipadapter, clip_vision,
    anima_diffusion, anima_text_encoder, anima_vae, anima_ipadapter, anima_control_model,
    resolution, fps, duration, seed, seed_mode, steps, cfg,
    denoise_start, denoise_peak, endpoint_strength, local_anchor_strength,
    curve, generation_strategy, low_vram, use_local_anchors,
    anima_ip_strength, anima_ip_cfg_scale, anima_ref_size, anima_ref_morph,
    anima_lllite_strength, anima_target_bias, anima_canny_low, anima_canny_high,
):
    if model_family == "Anima":
        return generate_sequence_anima_lllite(
            first, last, action_prompt, negative_prompt, style_notes,
            anima_diffusion, anima_text_encoder, anima_vae, anima_control_model, anima_ipadapter, resolution,
            fps, duration, seed, seed_mode, steps, min(5.0, float(cfg)),
            denoise_start, denoise_peak, anima_lllite_strength, anima_target_bias, anima_canny_low, anima_canny_high,
        )
    if model_family == "Anima-IP":
        return generate_sequence_anima(
            first, last, action_prompt, negative_prompt, style_notes,
            anima_diffusion, anima_text_encoder, anima_vae, anima_ipadapter,
            resolution, fps, duration, seed, seed_mode, steps, min(5.0, float(cfg)),
            denoise_start, denoise_peak, anima_ip_strength, anima_ip_cfg_scale, anima_ref_size, anima_ref_morph,
        )
    return generate_sequence(
        first, last, action_prompt, negative_prompt, style_notes, model_family,
        checkpoint, ipadapter, clip_vision, resolution, fps, duration, seed, seed_mode,
        steps, cfg, denoise_start, denoise_peak, endpoint_strength, local_anchor_strength,
        curve, generation_strategy, low_vram, use_local_anchors,
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
    pct = t * 100.0
    base = (
        f"{action_prompt.strip()}. This is animation in-between frame at {pct:.1f}% of the motion. "
        "Generate a physically plausible intermediate pose between the supplied temporal anchors. "
        "Keep the subject identity, facial structure, hairstyle, clothing, body proportions, camera, framing, "
        "background, lighting, palette and art style stable. Change only the elements required by the described action. "
        "Avoid teleporting, pose snapping, duplicate limbs, extra fingers, facial redesign, costume changes, "
        "background replacement, new objects, text, watermark, ghosting or double exposure."
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
    local_anchor_strength: float,
    low_vram: bool,
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
        frame_denoise = adaptive_denoise(t, float(denoise_start), float(denoise_peak), "Smootherstep")

        first_ref = upload_image(first_path)
        last_ref = upload_image(last_path)
        first_w = float(endpoint_strength) * (0.70 - 0.35 * t)
        last_w = float(endpoint_strength) * (0.20 + 0.35 * t)
        graph = api_graph_stable(
            checkpoint, ipadapter, clip_vision, first_ref, last_ref, first_ref,
            generation_prompt(action_prompt, t, style_notes), negative_prompt.strip(),
            base_seed, steps, cfg, frame_denoise, first_w, last_w, 0.06,
        )
        progress(0.1, desc="Submitting preview frame…")
        pid = submit(graph)
        frame = wait_for_image(pid, lambda: progress(0.25, desc="Generating preview frame…"))
        out = OUTPUT_DIR / f"FrameGeneratorV1_preview_{uuid.uuid4().hex[:10]}.png"
        shutil.copy2(frame, out)
        progress(1.0, desc="Preview complete")
        return str(out), (
            f"Preview position: {position:.1f}%\nSeed: {base_seed}\nModel: {model_family}\n"
            f"Resolution: {width}x{height}\nAdaptive denoise: {frame_denoise:.3f}\n"
            f"Global endpoint strength: {endpoint_strength:.2f}\nLocal anchor strength: {local_anchor_strength:.2f}"
        )
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
    local_anchor_strength: float,
    curve: str,
    generation_strategy: str,
    low_vram: bool,
    use_local_anchors: bool,
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

    plan = sequential_plan(total_frames)
    if not plan:
        raise gr.Error("No in-between frames were planned.")

    anchors: dict[int, Path] = {0: first_path, total_frames - 1: last_path}
    uploaded: dict[str, str] = {}

    def cached_upload(path: Path) -> str:
        key = str(path.resolve())
        if key not in uploaded:
            uploaded[key] = upload_image(path)
        return uploaded[key]

    first_ref = cached_upload(first_path)
    last_ref = cached_upload(last_path)
    generated: list[Path] = []

    progress(0.03, desc=f"Planning {total_frames} frames using {generation_strategy.lower()} temporal subdivision…")

    try:
        for step_number, (target_index, left_index, right_index) in enumerate(plan, start=1):
            if STOP_EVENT.is_set():
                interrupt_comfy()
                raise RuntimeError("Generation stopped by user.")

            left_path = anchors[left_index]
            right_path = anchors[right_index]
            t = target_index / (total_frames - 1)
            local_ratio = (target_index - left_index) / max(1, right_index - left_index)
            frame_denoise = adaptive_denoise(t, float(denoise_start), float(denoise_peak), curve)

            # Stability-first sequential generation: start from the immediately previous frame.
            # Keep both endpoints as low-weight global visual anchors; increase last-frame influence
            # gradually so the subject can move toward the supplied end pose without a sudden jump.
            first_ref = cached_upload(first_path)
            last_ref = cached_upload(last_path)
            previous_ref = cached_upload(left_path)
            u = time_curve(t, curve)
            first_w = float(endpoint_strength) * (0.70 - 0.35 * u)
            last_w = float(endpoint_strength) * (0.20 + 0.35 * u)
            temporal_w = float(local_anchor_strength)
            if target_index <= 1:
                temporal_w = min(0.05, temporal_w)

            stage_prompt = generation_prompt(action_prompt, t, style_notes)
            if seed_mode == "Fixed":
                frame_seed = base_seed
            else:
                frame_seed = (base_seed + target_index * 7919) % (2**31 - 1)

            graph = api_graph_stable(
                checkpoint, ipadapter, clip_vision, first_ref, last_ref, previous_ref,
                stage_prompt, negative_prompt.strip(), int(frame_seed), int(steps),
                float(cfg), float(frame_denoise), first_w, last_w, temporal_w,
            )
            pid = submit(graph)
            current_index = target_index + 1

            def cb():
                completed = step_number - 1
                progress(
                    0.05 + 0.90 * (completed / max(1, middle_count)),
                    desc=f"Generating frame {current_index}/{total_frames} — {t*100:.1f}% through action…",
                )

            tmp = wait_for_image(pid, progress_cb=cb)
            destination = render_dir / f"frame_{current_index:04d}.png"
            shutil.copy2(tmp, destination)
            anchors[target_index] = destination
            generated.append(destination)

        generated_sorted = sorted(generated, key=lambda p: int(p.stem.split('_')[-1]))
        all_frames = [first_out] + generated_sorted + [last_out]
        all_frames = sorted(all_frames, key=lambda p: int(p.stem.split('_')[-1]))
        all_zip = OUTPUT_DIR / f"FrameGeneratorV1_{job_id}_all_frames.zip"
        middle_zip = OUTPUT_DIR / f"FrameGeneratorV1_{job_id}_inbetween_only.zip"
        contact = OUTPUT_DIR / f"FrameGeneratorV1_{job_id}_contact_sheet.jpg"
        metadata = OUTPUT_DIR / f"FrameGeneratorV1_{job_id}_metadata.json"
        zip_frames(all_frames, all_zip)
        zip_frames(generated_sorted, middle_zip)
        save_contact_sheet(all_frames, contact)
        metadata.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "project": "Frame_Generator_V1",
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
                    "endpoint_strength": float(endpoint_strength),
                    "local_anchor_strength": float(local_anchor_strength),
                    "transition_curve": curve,
                    "generation_strategy": generation_strategy,
                    "local_temporal_anchors": bool(use_local_anchors),
                    "action_prompt": action_prompt,
                    "negative_prompt": negative_prompt,
                    "consistency_notes": style_notes,
                    "method": "stability-first sequential generation + global endpoint IP-Adapter references + local neighbor IP-Adapter references + latent interpolation bracket + adaptive denoise; PNG frames only",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        log_text = (
            f"Job: {job_id}\n"
            f"Strategy: {generation_strategy}\n"
            f"Total frames: {total_frames}\n"
            f"AI-generated middle frames: {middle_count}\n"
            f"FPS / duration: {fps} / {duration:.2f}s\n"
            f"Resolution: {width}x{height}\n"
            f"Seed: {base_seed} ({seed_mode})\n"
            f"Endpoint strength: {endpoint_strength:.2f}\n"
            f"Local temporal anchor strength: {local_anchor_strength:.2f} ({'ON' if use_local_anchors else 'OFF'})\n"
            f"Denoise schedule: {denoise_start:.2f} → {denoise_peak:.2f} (adaptive, low at endpoints / high in middle)\n"
            f"Curve: {curve}\n"
            f"Sequential previous-frame img2img anchor: active.\n"
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
    sd_required = ["IPAdapterAdvanced", "IPAdapterModelLoader", "CLIPVisionLoader"]
    anima_required = ["UNETLoader", "CLIPLoader", "VAELoader"]
    sd_missing = [n for n in sd_required if not node_available(n)]
    anima_missing = [n for n in anima_required if not node_available(n)]
    l_ok, l_missing, l_mode = anima_lllite_available()
    parts = []
    parts.append("✅ SD workflow ready" if not sd_missing else "⚠️ SD missing: " + ", ".join(sd_missing))
    parts.append("✅ Anima base ready" if not anima_missing else "⚠️ Anima base missing: " + ", ".join(anima_missing))
    parts.append(f"✅ Anima hybrid ready ({l_mode})" if l_ok else "⚠️ Anima hybrid unavailable: " + ", ".join(l_missing))
    return " | ".join(parts)


def app():
    models = scan_models()
    anima_models = scan_anima_models()
    anima_control_models = scan_anima_control_models()
    anima_control_default = guess_anima_control_model()
    family_default = str(CONFIG.get("default_model_family", "SD 1.5"))
    if family_default not in {"SD 1.5", "SDXL", "Anima"}:
        family_default = "SD 1.5"
    cp, ip, clip = guess_models(family_default if family_default in {"SD 1.5", "SDXL"} else "SD 1.5")
    a_diff, a_text, a_vae, a_ip = guess_anima_models()
    resolutions = ["512x512", "640x360", "640x384", "768x432", "768x512", "832x480", "1024x576", "1024x1024"]

    with gr.Blocks(title="Frame_Generator_V1") as demo:
        gr.Markdown(
            "# 🖼️ Frame_Generator_V1 — v8.4.2\n"
            "### First frame + last frame + action → **AI-generated PNG in-between frames**\n\n"
            "This is a frame generator, not a video generator. v8.4.1 uses the **Anima hybrid flow/img2img + IP-Adapter engine** while keeping SD 1.5/SDXL available."
        )

        with gr.Tab("Generate Frames"):
            with gr.Row():
                with gr.Column(scale=1):
                    first = gr.Image(type="filepath", sources=["upload", "clipboard"], label="First frame")
                    last = gr.Image(type="filepath", sources=["upload", "clipboard"], label="Last frame")
                with gr.Column(scale=1):
                    action_prompt = gr.Textbox(
                        label="Action / motion prompt", lines=6,
                        placeholder="Example: the girl slowly turns her head from left to right, raises her hand, hair moves gently in the wind",
                    )
                    style_notes = gr.Textbox(
                        label="Consistency instructions (optional)", lines=3,
                        value="same character design, same outfit, same camera and background, preserve anime line art and color palette",
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative prompt", lines=3,
                        value="deformed hands, extra fingers, extra limbs, duplicate character, inconsistent face, changed clothing, changed background, warped anatomy, teleporting pose, ghosting, double exposure, motion blur, text, watermark, blurry, low quality",
                    )
                    model_family = gr.Radio(["SD 1.5", "SDXL", "Anima"], value=family_default, label="Model family")

            with gr.Row(visible=family_default != "Anima") as sd_model_row:
                checkpoint = gr.Dropdown(choices=models["checkpoint"], value=cp or None, label="Checkpoint")
                ipadapter = gr.Dropdown(choices=models["ipadapter"], value=ip or None, label="IP-Adapter model")
                clip_vision = gr.Dropdown(choices=models["clip_vision"], value=clip or None, label="CLIP Vision")

            with gr.Row(visible=family_default == "Anima") as anima_model_row:
                anima_diffusion = gr.Dropdown(choices=anima_models["diffusion"], value=a_diff or None, label="Anima diffusion model")
                anima_text_encoder = gr.Dropdown(choices=anima_models["text_encoder"], value=a_text or None, label="Anima text encoder")
                anima_vae = gr.Dropdown(choices=anima_models["vae"], value=a_vae or None, label="Anima VAE")
                anima_ipadapter = gr.Dropdown(choices=(anima_models["ipadapter"] or [str(CONFIG.get("default_anima_ipadapter", "ip_adapter-Character_Reference-10.safetensors"))]), value=a_ip or str(CONFIG.get("default_anima_ipadapter", "ip_adapter-Character_Reference-10.safetensors")), label="Anima IP-Adapter (character reference)")
                anima_control_model = gr.Dropdown(choices=anima_control_models, value=anima_control_default or None, label="Legacy Anima LLLite control model (unused in v8.4.2)")

            with gr.Accordion("Anima controls", open=family_default == "Anima"):
                with gr.Row():
                    anima_lllite_strength = gr.Slider(0.0, 1.5, step=0.05, value=float(CONFIG.get("default_anima_lllite_strength", 0.65)), label="LLLite structure strength")
                    anima_target_bias = gr.Slider(0.0, 0.45, step=0.01, value=float(CONFIG.get("default_anima_lllite_target_bias", 0.18)), label="Endpoint target bias")
                    anima_canny_low = gr.Slider(0.0, 1.0, step=0.01, value=float(CONFIG.get("default_anima_canny_low", 0.17)), label="Canny low threshold")
                    anima_canny_high = gr.Slider(0.0, 1.0, step=0.01, value=float(CONFIG.get("default_anima_canny_high", 0.45)), label="Canny high threshold")
                with gr.Row():
                    anima_ip_strength = gr.Slider(0.0, 1.5, step=0.05, value=float(CONFIG.get("default_anima_ip_strength", 0.65)), label="Legacy Anima IP strength")
                    anima_ip_cfg_scale = gr.Slider(1.01, 5.0, step=0.05, value=float(CONFIG.get("default_anima_ip_cfg_scale", 2.0)), label="Legacy Anima IP CFG scale")
                    anima_ref_size = gr.Dropdown([384, 512, 640, 768], value=int(CONFIG.get("default_anima_ref_size", 512)), label="Legacy reference encoder size")
                    anima_ref_morph = gr.Slider(0.0, 0.35, step=0.01, value=float(CONFIG.get("default_anima_ref_morph", 0.15)), label="Legacy endpoint hint mix")
                gr.Markdown(
                    "**v8.4.2 Anima mode:** a first-to-last dense optical-flow guide is used as a full RGB img2img/reference image; the selected Anima IP-Adapter supplies character/style consistency. "
                    "The guide is temporary and is never exported. The legacy LLLite control field remains in the UI for compatibility; the active Anima engine is the hybrid flow/img2img + IP-Adapter path."
                )

            with gr.Accordion("Frame timing", open=True):
                with gr.Row():
                    resolution = gr.Dropdown(resolutions, value=str(CONFIG.get("default_resolution", "640x360")), label="Resolution")
                    fps = gr.Slider(6, 30, step=1, value=float(CONFIG.get("default_fps", 12)), label="FPS")
                    duration = gr.Slider(0.5, 10, step=0.5, value=float(CONFIG.get("default_duration", 2.0)), label="Duration (s)")
                frame_count = gr.Markdown(count_frames(CONFIG.get("default_fps", 12), CONFIG.get("default_duration", 2.0)))
                fps.change(count_frames, inputs=[fps, duration], outputs=frame_count)
                duration.change(count_frames, inputs=[fps, duration], outputs=frame_count)

            with gr.Accordion("Temporal consistency", open=True):
                with gr.Row():
                    generation_strategy = gr.Radio(
                        ["Sequential Stable"], value=str(CONFIG.get("default_generation_strategy", "Sequential Stable")),
                        label="Generation strategy",
                        info="Sequential generation uses the previous known frame as the img2img anchor while endpoint/reference conditioning supplies global direction."
                    )
                    curve = gr.Dropdown(
                        ["Linear", "Smoothstep", "Smootherstep", "Fast middle", "Fast start"],
                        value=str(CONFIG.get("default_curve", "Smoothstep")), label="Motion curve",
                    )
                with gr.Row():
                    endpoint_strength = gr.Slider(0.35, 1.30, step=0.05, value=float(CONFIG.get("default_endpoint_strength", 0.72)), label="Global endpoint reference strength")
                    local_anchor_strength = gr.Slider(0.0, 0.45, step=0.01, value=float(CONFIG.get("default_temporal_strength", 0.12)), label="Local temporal anchor strength")
                    denoise_start = gr.Slider(0.15, 0.55, step=0.01, value=float(CONFIG.get("default_denoise_start", 0.24)), label="Endpoint denoise")
                    denoise_peak = gr.Slider(0.35, 0.75, step=0.01, value=float(CONFIG.get("default_denoise_peak", 0.55)), label="Middle denoise")
                with gr.Row():
                    use_local_anchors = gr.Checkbox(value=True, label="Use nearby generated frames as temporal anchors")
                    gr.Markdown("For Anima, the same previous-frame img2img anchor is used, with the Anima IP-Adapter reference guiding character/style consistency.")

            with gr.Accordion("AI generation", open=True):
                with gr.Row():
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    seed_mode = gr.Radio(["Fixed", "Progressive"], value="Fixed", label="Seed strategy")
                    steps = gr.Slider(6, 50, step=1, value=int(CONFIG.get("default_steps", 16)), label="Steps")
                    cfg = gr.Slider(1, 12, step=0.5, value=float(CONFIG.get("default_cfg", 6.0)), label="CFG")
                with gr.Row():
                    low_vram = gr.Checkbox(value=bool(CONFIG.get("low_vram", True)), label="8 GB VRAM profile")
                gr.Markdown(
                    "For Anima, the UI caps CFG at 5 and uses an img2img start from the previous frame. "
                    "Official Anima documentation lists 30–50 steps and CFG 4–5 as a general generation range; the 8 GB profile is intentionally more conservative."
                )

            with gr.Accordion("Preview One Frame", open=False):
                preview_position = gr.Slider(1, 99, value=50, step=1, label="Preview position between endpoints (%)")
                preview_button = gr.Button("🔍 Generate One AI In-Between Frame", variant="secondary")
                preview_image = gr.Image(label="AI preview frame", type="filepath")
                preview_log = gr.Textbox(label="Preview log", lines=6)

            with gr.Row():
                generate_button = gr.Button("🚀 Generate PNG In-Between Sequence", variant="primary", size="lg")
                stop_button = gr.Button("⛔ Stop", variant="stop", size="lg")

            with gr.Row():
                preview_sheet = gr.Image(label="Generated frame sheet", type="filepath")
                all_zip = gr.File(label="All PNG frames (.zip)")
                middle_zip = gr.File(label="AI in-between frames only (.zip)")
            metadata_file = gr.File(label="Generation metadata (.json)")
            run_log = gr.Textbox(label="Run log", lines=10)

            common_preview_inputs = [
                first, last, action_prompt, negative_prompt, style_notes, model_family,
                checkpoint, ipadapter, clip_vision,
                anima_diffusion, anima_text_encoder, anima_vae, anima_ipadapter, anima_control_model,
                resolution, preview_position, seed, steps, cfg, denoise_start, denoise_peak,
                endpoint_strength, local_anchor_strength, anima_ip_strength, anima_ip_cfg_scale,
                anima_ref_size, anima_ref_morph, anima_lllite_strength, anima_target_bias, anima_canny_low, anima_canny_high,
            ]
            common_sequence_inputs = [
                first, last, action_prompt, negative_prompt, style_notes, model_family,
                checkpoint, ipadapter, clip_vision,
                anima_diffusion, anima_text_encoder, anima_vae, anima_ipadapter, anima_control_model,
                resolution, fps, duration, seed, seed_mode, steps, cfg,
                denoise_start, denoise_peak, endpoint_strength, local_anchor_strength,
                curve, generation_strategy, low_vram, use_local_anchors,
                anima_ip_strength, anima_ip_cfg_scale, anima_ref_size, anima_ref_morph, anima_lllite_strength, anima_target_bias, anima_canny_low, anima_canny_high,
            ]
            preview_button.click(generate_preview_dispatch, inputs=common_preview_inputs, outputs=[preview_image, preview_log])
            generate_button.click(generate_sequence_dispatch, inputs=common_sequence_inputs, outputs=[preview_sheet, all_zip, middle_zip, metadata_file, run_log])
            stop_button.click(stop_generation, outputs=run_log)

        with gr.Tab("Setup"):
            gr.Markdown(
                "## Connect to ComfyUI Desktop\n\n"
                "Frame_Generator_V1 connects through the local ComfyUI API and does not need the Desktop installation folder.\n\n"
                "1. Start ComfyUI Desktop.\n"
                "2. Leave the URL as `http://127.0.0.1:8188` unless you changed the port.\n"
                "3. Click **Save & Test Connection**.\n"
                "4. Click **Refresh model lists** after installing models or nodes."
            )
            comfy_url_box = gr.Textbox(value=comfy_url(), label="ComfyUI URL")
            with gr.Row():
                save_url_btn = gr.Button("Save & Test Connection", variant="primary")
                open_btn = gr.Button("Open ComfyUI")
                refresh_btn = gr.Button("Refresh model lists")
            status_box = gr.Textbox(label="Status", value=check_comfy()[1], lines=2)
            caps_box = gr.Textbox(label="Frame_Generator_V1 capabilities", value=capabilities_text(), lines=3)
            save_url_btn.click(save_comfy_url, inputs=comfy_url_box, outputs=status_box)
            open_btn.click(open_comfy, outputs=status_box)
            refresh_btn.click(
                refresh_all_model_lists,
                inputs=model_family,
                outputs=[checkpoint, ipadapter, clip_vision, anima_diffusion, anima_text_encoder, anima_vae, anima_ipadapter, anima_control_model, sd_model_row, anima_model_row],
            ).then(lambda: capabilities_text(), outputs=caps_box)
            model_family.change(
                sync_family,
                inputs=model_family,
                outputs=[checkpoint, ipadapter, clip_vision, anima_diffusion, anima_text_encoder, anima_vae, anima_ipadapter, anima_control_model, sd_model_row, anima_model_row],
            )
            gr.Markdown(
                "### SD 1.5 starting profile\n"
                "**640×360 / 12 FPS / 2 s / 16 steps / CFG 6** is a conservative starting point for an 8 GB GPU.\n\n"
                "### Anima v8 — Structure Control\n"
                "v8 recommends ComfyUI's native **Anima LLLite** path rather than the legacy Anima IP-Adapter path.\n\n"
                "Required Anima files:\n"
                "- `anima-base-v1.0.safetensors` → `models/diffusion_models/`\n"
                "- `qwen_3_06b_base.safetensors` → `models/text_encoders/`\n"
                "- `qwen_image_vae.safetensors` → `models/vae/`\n"
                "- `anima-lllite-any-test-like-v2.safetensors` → `models/model_patches/`\n\n"
                "The current ComfyUI core exposes `ModelPatchLoader` and `AnimaLLLiteApply` for this model-patch route. Update ComfyUI if those nodes are missing.\n\n"
                "### 8 GB note\n"
                "Start at **640×360**, **12–16 steps**, and **CFG 4–5** for Anima. v8 generates one image at a time and uses the previous frame as the identity/continuity anchor plus Canny-controlled Anima LLLite for structure.\n"
                "The Anima-LLLite weights remain under their own model license and are not included with Frame_Generator_V1."
            )

    return demo


if __name__ == "__main__":
    app().launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")), inbrowser=False, show_error=True)
