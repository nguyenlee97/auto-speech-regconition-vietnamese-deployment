# Vietnamese ASR â€” Google Colab Guide

This guide explains how to run the Vietnamese ZipFormer-30M RNNT model on Google Colab using the provided notebook and how to expose a public HTTP API for testing.

## Files
- `colab/vi_asr_colab.ipynb` â€” endâ€‘toâ€‘end notebook: install deps, run a quick inference, start FastAPI, and expose a public URL via Cloudflared.

## Open the Notebook in Colab
You can use any of the following methods:

1) If the repo is on GitHub, open (replace OWNER/REPO/BRANCH as needed):
   - `https://colab.research.google.com/github/OWNER/REPO/blob/BRANCH/colab/vi_asr_colab.ipynb`

2) From Colab UI: File â†’ Open notebook â†’ GitHub tab â†’ paste your repo URL and open `colab/vi_asr_colab.ipynb`.

3) Upload the notebook directly: File â†’ Upload notebook â†’ select `colab/vi_asr_colab.ipynb` from your local clone.

If you open the notebook outside the repo root (typical when using the GitHub link), run this bootstrap cell first in Colab to clone the repo and change directory:\n\n```\n!git clone https://github.com/nguyenlee97/auto-speech-regconition-vietnamese-deployment.git \
    /content/auto-speech-regconition-vietnamese-deployment\n%cd /content/auto-speech-regconition-vietnamese-deployment\n```\nAfter this, run the notebook cells in order.

## Run Order and What Each Cell Does
Run the cells from top to bottom:

1) Environment setup
   - Installs `ffmpeg`, Python deps, and Cloudflared. If your runtime is Python 3.10, the pinned wheels in `requirements.txt` are used. If you are on Python 3.12+, see the note below.

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
- If Colab reports dependency conflicts, use â€œRuntime â†’ Restart runtimeâ€ and re-run the setup cell.
- If the notebook canâ€™t find model files, ensure you are in the repo root or clone the repo in a cell and `cd` into it.




### Python 3.12+ on Colab
equirements.txt pins CPU wheels for Python 3.10. If your Colab shows Python 3.12+ in the setup cell, you have two options:

- Preferred: Runtime → Change runtime type → set Python version to 3.10 (if available), then rerun cells.

- Alternative (stay on 3.12+): Install compatible wheels and use the ONNX path only.
  Run this in a new cell before importing model:

  `
  !pip install --upgrade pip wheel setuptools
  !pip install --index-url https://download.pytorch.org/whl/cpu \
      torch==2.5.1+cpu torchaudio==2.5.1+cpu
  !pip install sherpa-onnx fastapi uvicorn[standard] python-multipart requests sentencepiece numpy==1.26.4
  `

  Then patch model.py so it skips optional k2/sherpa imports (not needed for Vietnamese ONNX):
  `python
  import io
  p = 'model.py'
  s = open(p, 'r', encoding='utf-8').read()
  if 'from __future__ import annotations' not in s:
      s = 'from __future__ import annotations\n' + s
  s = s.replace('import k2  # noqa', 'try:\n    import k2  # noqa\nexcept Exception:\n    k2 = None')
  s = s.replace('import sherpa', 'try:\n    import sherpa\nexcept Exception:\n    sherpa = None')
  open(p, 'w', encoding='utf-8').write(s)
  print('Patched model.py for Colab (Py3.12+)')
  `

  Proceed with the rest of the notebook (quick inference and API server). This uses sherpa-onnx with the Vietnamese ONNX models.

