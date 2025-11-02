# Vietnamese ASR API — Easy Deployment Guide

This repo includes a FastAPI server (`api_server.py`) that exposes a simple HTTP API for Vietnamese speech recognition using `sherpa-onnx` ZipFormer models (int8 for fast CPU inference).

## Quick Start (Docker Compose)

Prereqs: Linux VPS, Docker Engine + Compose plugin, ports `8080` (HTTP) or `80/443` (HTTPS + Caddy) open.

1) Copy and edit env file

```bash
cp .env.example .env
# Edit .env to set API_KEY and DOMAIN
```

2) Build and start

```bash
# Start API on http://localhost:8080
docker compose build
docker compose up -d

# Optionally add HTTPS via Caddy (auto TLS on DOMAIN)
docker compose -f docker-compose.yml -f docker-compose.caddy.yml up -d
```

3) Health

```bash
curl -sf http://localhost:8080/healthz
curl -sf http://localhost:8080/readyz
```

4) Transcribe (examples)

- Multipart upload
```bash
curl -X POST "http://localhost:8080/v1/transcribe?decoding_method=modified_beam_search&num_active_paths=15" \
  -H "Authorization: Bearer $(grep ^API_KEY .env | cut -d= -f2)" \
  -F "file=@test_wavs/vietnamese/0.wav"
```

- JSON URL
```bash
curl -X POST http://localhost:8080/v1/transcribe \
  -H "Authorization: Bearer $(grep ^API_KEY .env | cut -d= -f2)" \
  -H "Content-Type: application/json" \
  -d '{"audio_url":"https://example.com/sample.wav","decoding_method":"greedy_search","num_active_paths":8}'
```

- JSON Base64
```bash
base64 -w0 test_wavs/vietnamese/0.wav > /tmp/a.b64
curl -X POST http://localhost:8080/v1/transcribe \
  -H "Authorization: Bearer $(grep ^API_KEY .env | cut -d= -f2)" \
  -H "Content-Type: application/json" \
  -d "{\"audio_base64\":\"$(cat /tmp/a.b64)\"}"
```

## What Compose Files Do

- `docker-compose.yml`
  - Builds `vi-asr` image from this repo
  - Exposes `${HOST_PORT:-8080} -> 8000`
  - Healthcheck on `/readyz` (container has curl)
  - Reads `.env` for model and security settings

- `docker-compose.caddy.yml` (optional)
  - Adds `caddy` reverse proxy with automatic Let’s Encrypt TLS
  - Requires `DOMAIN` in `.env` to be public and pointing to the VPS

- `Caddyfile`
  - Proxies `${DOMAIN}` to `vi-asr:8000`, adds CORS headers

- `deploy/nginx.conf` (optional)
  - Example Nginx site if you prefer Nginx + certbot instead of Caddy

## Environment Variables (from .env)

- `HOST_PORT` — host port for API (default 8080)
- `REQUIRE_API_KEY` — `true|false` to enforce bearer auth
- `API_KEY` — secret used when auth is enabled
- `MODEL_REPO_ID` — default `hynt/sherpa-onnx-zipformer-vi-int8-2025-10-16`
- `DECODING_METHOD` — `modified_beam_search|greedy_search`
- `NUM_ACTIVE_PATHS` — default 15
- `MAX_DURATION_SEC` — default 60
- `DOMAIN` — domain used by Caddy for automatic TLS

## Reverse Proxy — Nginx (Alternative)

1) Put `deploy/nginx.conf` to `/etc/nginx/sites-available/vi-asr.conf` and symlink to `sites-enabled`:
```bash
sudo ln -s /etc/nginx/sites-available/vi-asr.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```
2) Set up TLS with certbot if needed.

## Production Tips

- Keep `REQUIRE_API_KEY=true` and use HTTPS (Caddy or Nginx + certbot).
- Limit audio length with `MAX_DURATION_SEC` (default 60s) and proxy `client_max_body_size`.
- For more traffic, scale horizontally by running multiple `vi-asr` instances and load balance at the proxy.
- Use the int8 ONNX model on CPU for best latency.

## Local Dev (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
UVICORN_PORT=8000 REQUIRE_API_KEY=false uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
```
