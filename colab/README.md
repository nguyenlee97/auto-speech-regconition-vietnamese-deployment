# Vietnamese ASR — Google Colab Guide

This guide explains how to run the Vietnamese ZipFormer-30M RNNT model on Google Colab using the provided notebook and how to expose a public HTTP API for testing.

## Files
- `colab/vi_asr_colab.ipynb` — end‑to‑end notebook: install deps, run a quick inference, start FastAPI, and expose a public URL via Cloudflared.

## Open the Notebook in Colab
You can use any of the following methods:

1) If the repo is on GitHub, open (replace OWNER/REPO/BRANCH as needed):
   - `https://colab.research.google.com/github/OWNER/REPO/blob/BRANCH/colab/vi_asr_colab.ipynb`

2) From Colab UI: File → Open notebook → GitHub tab → paste your repo URL and open `colab/vi_asr_colab.ipynb`.

3) Upload the notebook directly: File → Upload notebook → select `colab/vi_asr_colab.ipynb` from your local clone.

If you open the notebook outside the repo root, the cells will instruct you to either cd into the repo directory or clone it so that model files and test audio are available.

## Run Order and What Each Cell Does
Run the cells from top to bottom:

1) Environment setup
   - Installs `ffmpeg`, pinned Python wheels (`torch` CPU, `k2/sherpa`, `sherpa-onnx`, and the FastAPI stack), and downloads Cloudflared (for a public tunnel).

2) Verify setup
   - Prints Python/platform, checks that `model.py`, ONNX artifacts (`encoder-*.onnx`, `decoder-*.onnx`, `joiner-*.onnx`), and `config.json` are present, and lists sample wavs in `test_wavs/vietnamese/`.

3) Quick local inference
   - Loads the default model `hynt/sherpa-onnx-zipformer-vi-int8-2025-10-16` and decodes `test_wavs/vietnamese/0.wav` to verify everything works.

4) Start FastAPI (background)
   - Launches `api_server.py` on `http://127.0.0.1:8000` inside Colab.
   - You can adjust environment variables before launching:
     - `MODEL_REPO_ID` (default: int8 Vietnamese model)
     - `DECODING_METHOD` (`modified_beam_search` or `greedy_search`)
     - `NUM_ACTIVE_PATHS` (default 15)
     - `MAX_DURATION_SEC` (default 60)
     - `REQUIRE_API_KEY` (`true|false`, default `false`)
     - `API_KEY` (if `REQUIRE_API_KEY=true`)

5) Expose a public URL via Cloudflared
   - Starts a temporary tunnel and prints a URL like `https://xxxxx.trycloudflare.com`.
   - Endpoints:
     - `GET /healthz` (liveness)
     - `GET /readyz` (model warm/ready)
     - `POST /v1/transcribe` (multipart or JSON)

6) Call the API (examples)
   - Demonstrates multipart upload and base64 JSON calls from within the notebook.

## External Testing from Your Machine
Replace `PUBLIC_URL` with the `trycloudflare.com` URL printed by the notebook.

- Multipart upload:
```
curl -X POST "PUBLIC_URL/v1/transcribe" \
  -F "file=@test_wavs/vietnamese/0.wav"
```

- JSON (base64):
```
base64 -w0 test_wavs/vietnamese/0.wav > /tmp/a.b64
curl -X POST "PUBLIC_URL/v1/transcribe" \
  -H "Content-Type: application/json" \
  -d "{\"audio_base64\":\"$(cat /tmp/a.b64)\"}"
```

If you enabled authentication, add the header:

```
-H "Authorization: Bearer YOUR_API_KEY"
```

## Notes and Tips
- Default model: `hynt/sherpa-onnx-zipformer-vi-int8-2025-10-16` (fast CPU inference).
- Cloudflared URLs are ephemeral and for testing only. For production, consider the Docker/Caddy approach in `DEPLOYMENT.md`.
- If Colab reports dependency conflicts, use “Runtime → Restart runtime” and re-run the setup cell.
- If the notebook can’t find model files, ensure you are in the repo root or clone the repo in a cell and `cd` into it.

