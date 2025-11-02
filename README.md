---
title: Automatic Speech Recognition
emoji: 🌍
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 5.25.2
python_version: 3.10.0
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Colab Notebook

- Use the Colab notebook in `colab/vi_asr_colab.ipynb` to:
  - Install dependencies and ffmpeg on Colab
  - Run a quick local inference with sample Vietnamese audio
  - Launch the FastAPI server and expose a public URL for API testing

- Quick start:
  1. Open `colab/vi_asr_colab.ipynb` in Google Colab (via GitHub link or upload)
  2. Run cells 1→6 in order
  3. Copy the printed `trycloudflare.com` URL to call `/v1/transcribe`

- See `colab/README.md` for detailed, step‑by‑step instructions and example `curl` commands.
