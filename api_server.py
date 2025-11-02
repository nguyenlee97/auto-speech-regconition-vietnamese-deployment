import base64
import io
import os
import tempfile
import time
import wave
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Header, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask
from starlette.middleware.cors import CORSMiddleware

from model import (
    decode,
    get_pretrained_model,
    sample_rate,
)

import subprocess
import shutil
import urllib.request


APP_NAME = "vi-asr"


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes")


DEFAULT_REPO_ID = _env_str(
    "MODEL_REPO_ID", "hynt/sherpa-onnx-zipformer-vi-int8-2025-10-16"
)
DEFAULT_DECODING_METHOD = _env_str("DECODING_METHOD", "modified_beam_search")
DEFAULT_NUM_ACTIVE_PATHS = _env_int("NUM_ACTIVE_PATHS", 15)
MAX_DURATION_SEC = _env_int("MAX_DURATION_SEC", 60)
REQUIRE_API_KEY = _env_bool("REQUIRE_API_KEY", False)
API_KEY = os.getenv("API_KEY", "")


def _ffmpeg_convert_to_wav(in_path: str) -> str:
    out_path = f"{in_path}.wav"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                in_path,
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                out_path,
                "-y",
            ],
            check=True,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg not found in PATH")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"ffmpeg failed: {e}")
    return out_path


def _read_wav_duration(path: str) -> float:
    with wave.open(path, "rb") as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)


async def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "audio")[1]
    fd, tmp_path = tempfile.mkstemp(prefix="upload_", suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    await upload.close()
    return tmp_path


def _save_bytes_to_temp(data: bytes, suffix: str = "") -> str:
    fd, tmp_path = tempfile.mkstemp(prefix="audio_", suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(data)
    return tmp_path


def _fetch_url_to_temp(url: str) -> str:
    fd, tmp_path = tempfile.mkstemp(prefix="url_", suffix="")
    os.close(fd)
    try:
        urllib.request.urlretrieve(url, tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")
    return tmp_path


def _require_auth(authorization: Optional[str]) -> None:
    if not REQUIRE_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warm_model() -> None:
    try:
        _ = get_pretrained_model(
            DEFAULT_REPO_ID,
            decoding_method=DEFAULT_DECODING_METHOD,
            num_active_paths=DEFAULT_NUM_ACTIVE_PATHS,
        )
    except Exception as e:
        # Defer crash to readiness unless strictly required
        print(f"[startup] Failed to warm model: {e}")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    try:
        _ = get_pretrained_model(
            DEFAULT_REPO_ID,
            decoding_method=DEFAULT_DECODING_METHOD,
            num_active_paths=DEFAULT_NUM_ACTIVE_PATHS,
        )
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not ready: {e}")


@app.post("/v1/transcribe")
async def transcribe(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    _require_auth(authorization)

    content_type = request.headers.get("content-type", "")
    src = "unknown"
    cleanup_paths = []

    decoding_method = request.query_params.get("decoding_method", DEFAULT_DECODING_METHOD)
    try:
        num_active_paths = int(request.query_params.get("num_active_paths", DEFAULT_NUM_ACTIVE_PATHS))
    except ValueError:
        num_active_paths = DEFAULT_NUM_ACTIVE_PATHS
    repo_id = request.query_params.get("repo_id", DEFAULT_REPO_ID)

    in_path = None

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            file: UploadFile = form.get("file")
            if not isinstance(file, UploadFile):
                raise HTTPException(status_code=400, detail="Missing file in form-data under key 'file'")
            in_path = await _save_upload_to_temp(file)
            src = "upload"
        elif "application/json" in content_type:
            data = await request.json()
            if not isinstance(data, dict):
                raise HTTPException(status_code=400, detail="Invalid JSON body")
            if "audio_url" in data:
                in_path = _fetch_url_to_temp(str(data["audio_url"]))
                src = "url"
            elif "audio_base64" in data:
                try:
                    audio_bytes = base64.b64decode(data["audio_base64"], validate=True)
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid base64: unable to decode")
                in_path = _save_bytes_to_temp(audio_bytes)
                src = "base64"
            else:
                raise HTTPException(status_code=400, detail="Provide 'audio_url' or 'audio_base64' in JSON body, or send multipart with 'file'")
            # Override options if provided in JSON
            decoding_method = data.get("decoding_method", decoding_method)
            num_active_paths = int(data.get("num_active_paths", num_active_paths))
            repo_id = data.get("repo_id", repo_id)
        else:
            raise HTTPException(status_code=415, detail="Unsupported Content-Type. Use multipart/form-data or application/json")

        cleanup_paths.append(in_path)
        wav_path = _ffmpeg_convert_to_wav(in_path)
        cleanup_paths.append(wav_path)

        duration = _read_wav_duration(wav_path)
        if duration > MAX_DURATION_SEC:
            raise HTTPException(status_code=413, detail=f"Audio too long: {duration:.2f}s > {MAX_DURATION_SEC}s")

        start = time.time()
        recognizer = get_pretrained_model(
            repo_id,
            decoding_method=decoding_method,
            num_active_paths=num_active_paths,
        )
        text = decode(recognizer, wav_path)
        end = time.time()

        inference_sec = end - start
        rtf = inference_sec / max(duration, 1e-6)

        resp = {
            "text": text,
            "duration_sec": round(duration, 3),
            "inference_sec": round(inference_sec, 3),
            "rtf": round(rtf, 3),
            "model_repo": repo_id,
            "decoding_method": decoding_method,
            "num_active_paths": int(num_active_paths),
            "source": src,
            "language": "vi",
        }

        def _cleanup(paths):
            for p in paths:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

        return JSONResponse(resp, background=BackgroundTask(_cleanup, cleanup_paths))
    except HTTPException:
        # pass through
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = _env_int("UVICORN_PORT", 8000)
    # Limit torch/blas threads for predictable CPU use
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, workers=1)