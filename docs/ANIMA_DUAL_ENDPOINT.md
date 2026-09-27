# Anima Dual-Endpoint Conditioning (v8.5)

## Problem addressed

Earlier versions built a temporary blended reference image. In the observed run, most middle frames stayed close to the first endpoint and the final endpoint had weak semantic influence.

## v8.5 approach

Each generated frame uses three independent sources:

1. Previous generated frame -> VAE img2img latent anchor for local temporal continuity.
2. Original first endpoint -> Anima IP-Adapter reference, with strength proportional to `(1 - t)`.
3. Original last endpoint -> Anima IP-Adapter reference, with strength proportional to `t`.

The first and last endpoint images are never pixel-crossfaded for the IP-Adapter. Their semantic guidance is applied separately.

At `t=0.25` with total endpoint strength 0.72:
- first endpoint strength ≈ 0.54
- last endpoint strength ≈ 0.18

At `t=0.50`:
- first ≈ 0.36
- last ≈ 0.36

At `t=0.75`:
- first ≈ 0.18
- last ≈ 0.54

The sampler still starts from the previous frame, with denoise controlled by the temporal denoise curve.

This keeps the image-sequence-first design: every middle result is an individual PNG.
