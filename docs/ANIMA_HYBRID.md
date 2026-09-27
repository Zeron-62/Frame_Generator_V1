# Anima Hybrid Conditioning (v8.4)

v8.4 changes the Anima structure path to a semantic full-image hybrid instead of a Canny-only LLLite guide.

## Pipeline

1. Compute a dense optical-flow trajectory between the first and last endpoints.
2. Warp the first endpoint toward the last endpoint at the target time `t`.
3. Use the resulting full RGB guide as the `VAEEncode` latent initialization.
4. Feed the same RGB guide to Anima IP-Adapter as the reference image.
5. Generate one PNG with a conservative img2img denoise value.

The guide is temporary and is never exported as a final frame.

## Why

The earlier LLLite-only path converted the guide to sparse edges and started from a fresh empty latent. In the observed test, this caused the generated image to lose most of the original scene semantics and converge on simplified glowing shapes. v8.4 keeps the actual scene pixels in the generation path.

## 8 GB profile

Suggested starting point:

- 640x360
- 12 FPS
- 2 seconds
- 16 steps
- CFG 4
- denoise 0.30 near endpoints, up to 0.50 in the middle
- Anima IP-Adapter strength 0.45
- IP CFG 1.6
- fixed seed for A/B testing

Keep the existing SD 1.5 path as the regression baseline.
