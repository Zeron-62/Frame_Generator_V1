# Base44 dev environment — Frame_Generator_V1 (FrameForge)

## What this app is
A single-file **Gradio** web UI (`app.py`, ~47 KB) for AI in-between frame generation.
It does NOT run any AI model itself — it drives a separate **ComfyUI** backend over its
local HTTP API (default `http://127.0.0.1:8188`, configured in `config.json`).

- Stack: Python 3 + Gradio 5/6 + `requests` + Pillow (see `requirements.txt`).
- No database, no migrations, no seeds.
- No environment variables are required to boot — all config lives in `config.json`.
  The app boots and shows its UI even when ComfyUI is unreachable; generation just fails.

## ComfyUI backend (external, user-provided)
ComfyUI cannot run in this sandbox (needs a GPU + multi-GB model weights) and the
user's local ComfyUI (`127.0.0.1:8188`) is not reachable from the container. So:
- The preview shows the full UI and the "Setup" tab reports "Cannot reach ComfyUI".
- To make generation work, the user must point the app at a ComfyUI instance that is
  reachable from the sandbox (a public URL), set via the Setup tab or `config.json`.
- ComfyUI's API is unauthenticated — no credentials are needed.

## Running it
```
docker compose -f docker-compose.base44.yml up -d --build
```
- Web entry point is on host port **3000** (mapped to the container's Gradio port 7860).
- `app.py`'s launch line was changed only to bind `0.0.0.0` and disable `inbrowser`
  (no browser exists in the container). No app logic was changed.

## Editing / live reload
Gradio has no built-in file watcher, so source edits are NOT hot-reloaded. After
changing `app.py`, either call `reload_preview` or `docker compose restart web`,
then check `docker compose logs web`.

## Verify it works
- `curl -sS http://localhost:3000/` returns the Gradio HTML shell.
- `docker compose logs web` should show `Running on local URL: http://0.0.0.0:7860`.
