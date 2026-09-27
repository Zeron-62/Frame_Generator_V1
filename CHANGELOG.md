# Changelog

## v8.5

- Replaced single blended Anima IP-Adapter reference with separate first/last endpoint references.
- First endpoint IP-Adapter strength decreases with time; last endpoint strength increases with time.
- Previous generated frame remains the VAE img2img latent anchor for local temporal continuity.
- Removed dependency on a temporary first/last pixel-blended reference for the main Anima sequence path.


## v8.4.2

- Fixed the Anima IP-Adapter selection path so the active Anima dispatcher receives the UI-selected checkpoint.
- Added a robust configured-checkpoint fallback when the ComfyUI node API returns no model choices.
- The default character-reference checkpoint is `ip_adapter-Character_Reference-10.safetensors`.
- Updated UI/version strings to identify the active hybrid engine.


## v8.4.1

- Fixed Anima IP-Adapter selection being lost when the UI dropdown is empty or stale.
- Active Anima generation now passes the UI-selected IP-Adapter through preview and sequence dispatchers.
- Added a robust fallback to the configured character-reference checkpoint.
- Updated metadata and UI text to reflect the active hybrid engine.


## v8.4

- Replaced the default Anima Canny/LLLite generation path with a full-RGB optical-flow guide + img2img latent initialization + Anima IP-Adapter hybrid.
- Removed dependence on the Anima LLLite model patch for the default Anima structure route.
- Retained the existing 8 GB-friendly sequential frame architecture and safety validation.
- Kept legacy LLLite UI fields for compatibility while making the hybrid path the active Anima route.

## v8.3 - Endpoint-aware structural guide
- Replaced recursive previous-frame LLLite control with optical-flow-warped first-to-last structural guides.
- Keeps fresh EmptyLatentImage generation and automatic black-frame recovery.
- Added OpenCV dependency for local dense optical flow; falls back to the source frame if unavailable.

# Changelog

## v8.2.1 - LLLite sequence hotfix

- Fixed a frame validation bug that could reject every generated frame when NumPy was unavailable, causing the sequence fallback to duplicate the previous frame.
- Removed the NumPy dependency from frame-usability checks.
- Wired adaptive denoise into the Anima LLLite KSampler instead of hard-coding denoise to 1.0.
- Keeps the v8.2 fresh-latent + Canny/inverted-control architecture unchanged for a controlled retest.


## v8.1
- Fixed current ComfyUI `UNETLoader` API validation by explicitly sending `weight_dtype: default` in Anima workflows.
# Changelog

## v8.2 — LLLite stability correction
- Aligned the Anima LLLite graph with the current official control-to-image pattern: Canny → ImageInvert → LLLite, with a fresh EmptyLatentImage.
- Removed recursive previous-frame img2img latent initialization from the Anima LLLite path.
- Added last-known-good frame control and automatic near-black frame detection/retry.
- Lowered the default LLLite strength and Canny thresholds for the 8 GB test profile.


## v8.0.0 - Experimental Anima Structure Control

- Added recommended Anima LLLite structure-control backend using ComfyUI's native `ModelPatchLoader` + `AnimaLLLiteApply` path.
- Added temporal structure-guide generation from the previous frame toward the supplied last frame.
- Added Canny-based geometry control before the Anima LLLite patch.
- Kept previous-frame img2img as the primary identity/continuity anchor.
- Kept 16:9 presets such as 640x360 as the default direction for animation-frame work.
- Retained the previous Anima IP-Adapter path in code for compatibility, but it is no longer the default Anima generation path.
- Added `docs/ANIMA_LLLITE.md` with current model and workflow references.
