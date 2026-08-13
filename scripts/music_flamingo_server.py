#!/usr/bin/env python3
"""Minimal Music Flamingo server speaking Side-Step's local-provider protocol.

Side-Step's Music Flamingo provider, when pointed at a loopback/private
address, calls::

    POST {server_url}/caption
    Content-Type: multipart/form-data
      file=<audio file>
      prompt=<instruction text>

and expects a JSON object back.  Recognised keys (see
``metadata_provider_music_flamingo._normalize_json_keys``)::

    caption, genres, bpm, key_scale, timesignature,
    vocal_language, is_instrumental

Anything unparseable falls back to being treated as a caption string, so
returning well-formed JSON matters.

Usage:
    python scripts/music_flamingo_server.py --model /path/to/music-flamingo
    python scripts/music_flamingo_server.py --model /path/to/model --port 8100

Then set the Music Flamingo URL in Side-Step to::

    http://127.0.0.1:8100

Requires: fastapi, uvicorn, python-multipart (plus the model's own deps).
"""

# NOTE: deliberately no ``from __future__ import annotations`` here.
# PEP 563 turns the endpoint annotations into strings, and FastAPI then
# cannot resolve ``UploadFile`` (imported inside build_app, so not a
# module global) — every upload fails with a PydanticUserError.

import argparse
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("music_flamingo_server")

# Populated by load_model() at startup.
_MODEL: Any = None
_PROCESSOR: Any = None
_DEVICE: str = "cuda"


# ── Model loading ──────────────────────────────────────────────────

def load_model(model_path: str, dtype: str) -> None:
    """Load the model once at startup and cache it module-level.

    Follows the nvidia/music-flamingo-2601-hf model card: a dedicated
    ``MusicFlamingoForConditionalGeneration`` class plus ``AutoProcessor``,
    with ``device_map="auto"`` handling placement.
    """
    global _MODEL, _PROCESSOR, _DEVICE

    import torch
    from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration

    kwargs: Dict[str, Any] = {"device_map": "auto"}
    if dtype != "auto":
        kwargs["dtype"] = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[dtype]

    logger.info("Loading Music Flamingo from %s (dtype=%s)", model_path, dtype)
    _PROCESSOR = AutoProcessor.from_pretrained(model_path)
    _MODEL = MusicFlamingoForConditionalGeneration.from_pretrained(model_path, **kwargs)
    _MODEL.eval()
    _DEVICE = str(_MODEL.device)
    logger.info("Model ready: %s on %s", type(_MODEL).__name__, _DEVICE)


def run_inference(audio_path: str, prompt: str, max_new_tokens: int) -> str:
    """Run one audio+prompt generation and return the raw text response."""
    import torch

    if _MODEL is None:
        raise RuntimeError("Model is not loaded")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": audio_path},
            ],
        }
    ]

    inputs = _PROCESSOR.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(_MODEL.device)
    inputs["input_features"] = inputs["input_features"].to(_MODEL.dtype)

    with torch.no_grad():
        outputs = _MODEL.generate(**inputs, max_new_tokens=max_new_tokens)

    # Slice off the prompt tokens so only the reply is decoded.
    reply = outputs[:, inputs.input_ids.shape[1]:]
    return _PROCESSOR.batch_decode(reply, skip_special_tokens=True)[0]


# ── Response shaping ───────────────────────────────────────────────

# Music Flamingo caps audio at 20 minutes (30s windows). The model card
# says longer input is truncated, but the implementation instead indexes
# out of bounds in the audio positional embedding and fires a CUDA
# device-side assert — which corrupts the context for the whole process.
# Truncate ourselves, with a margin, rather than trusting that.
_MAX_AUDIO_SECONDS = 19 * 60

# The model consumes audio in 30s windows. Lengths landing exactly on a
# window boundary overflow the positional-embedding table by one.
_WINDOW_SECONDS = 30
_BOUNDARY_EPSILON = 0.05
_BOUNDARY_PAD_SECONDS = 0.5


def _pad_with_silence(path: str, samplerate: int) -> str:
    """Append a little silence and return the new temp file's path."""
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    pad = np.zeros((int(_BOUNDARY_PAD_SECONDS * sr), data.shape[1]), dtype="float32")
    padded = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    padded.close()
    sf.write(padded.name, np.concatenate([data, pad], axis=0), sr)
    return padded.name


def prepare_audio(path: str) -> str:
    """Return a path safe to feed the model, truncating over-long audio.

    Writes a shortened copy when needed; the original is left alone.
    """
    import soundfile as sf

    try:
        info = sf.info(path)
        duration = float(info.frames) / float(info.samplerate or 1)
    except Exception as exc:
        logger.warning("Could not probe duration (%s); passing through", exc)
        return path

    if duration <= _MAX_AUDIO_SECONDS:
        # Audio whose length lands exactly on a 30s window boundary trips an
        # off-by-one in the audio positional embedding (index == table size),
        # firing a CUDA device-side assert. A little silence moves it off the
        # boundary and is inaudible to the model's description.
        remainder = duration % _WINDOW_SECONDS
        if remainder < _BOUNDARY_EPSILON or remainder > (_WINDOW_SECONDS - _BOUNDARY_EPSILON):
            logger.warning(
                "Duration %.3fs sits on a %ds window boundary — padding %.1fs "
                "of silence to avoid the audio-embedding off-by-one",
                duration, _WINDOW_SECONDS, _BOUNDARY_PAD_SECONDS,
            )
            return _pad_with_silence(path, info.samplerate)
        return path

    keep_frames = int(_MAX_AUDIO_SECONDS * info.samplerate)
    logger.warning(
        "Audio is %.1f min, over the %.0f min cap — truncating for inference",
        duration / 60.0, _MAX_AUDIO_SECONDS / 60.0,
    )
    data, sr = sf.read(path, frames=keep_frames, dtype="float32", always_2d=True)
    trimmed = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    trimmed.close()
    sf.write(trimmed.name, data, sr)
    return trimmed.name


def _is_cuda_assert(exc: BaseException) -> bool:
    """A device-side assert leaves the CUDA context unusable process-wide."""
    return "device-side assert" in str(exc).lower()


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)

_ALLOWED_KEYS = (
    "caption", "genres", "bpm", "key_scale",
    "timesignature", "vocal_language", "is_instrumental",
)


def shape_response(raw_text: str) -> Dict[str, Any]:
    """Coerce the model's reply into the JSON object Side-Step expects.

    The prompt asks for JSON, but models wrap it in prose or code fences.
    Pull out the first JSON object if there is one; otherwise treat the
    whole reply as a caption.
    """
    text = (raw_text or "").strip()
    if not text:
        return {"caption": ""}

    fenced = text
    if "```" in fenced:
        parts = [p for p in fenced.split("```") if p.strip()]
        for part in parts:
            cleaned = part.strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.startswith("{"):
                fenced = cleaned
                break

    match = _JSON_BLOCK.search(fenced)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                out = {k: v for k, v in obj.items() if k in _ALLOWED_KEYS and v not in (None, "")}
                if out:
                    return out
        except json.JSONDecodeError:
            logger.debug("Reply contained a JSON-like block that did not parse")

    return {"caption": text}


# ── HTTP app ───────────────────────────────────────────────────────

def build_app(max_new_tokens: int):
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Music Flamingo (Side-Step local provider)")

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"ok": _MODEL is not None, "device": _DEVICE}

    @app.post("/caption")
    async def caption(
        file: UploadFile = File(...),
        prompt: str = Form(""),
    ):
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        audio_path = tmp.name
        try:
            with tmp:
                shutil.copyfileobj(file.file, tmp)
            logger.info("caption: %s (%s)", file.filename, prompt[:60])
            audio_path = prepare_audio(tmp.name)
            raw = run_inference(audio_path, prompt, max_new_tokens)
            payload = shape_response(raw)
            logger.info("  -> %s", json.dumps(payload)[:200])
            return JSONResponse(payload)
        except Exception as exc:
            logger.exception("Inference failed for %s", file.filename)
            if _is_cuda_assert(exc):
                # The CUDA context is now poisoned: every later request in
                # this process fails too, turning one bad file into a run
                # of bogus failures. Die so the supervisor restarts clean.
                logger.critical(
                    "CUDA context corrupted by %s — exiting so the server "
                    "restarts. Remaining files would otherwise all fail.",
                    file.filename,
                )
                import threading
                threading.Timer(0.5, lambda: os._exit(70)).start()
                return JSONResponse(
                    {"error": f"CUDA assert on {file.filename}; server restarting"},
                    status_code=503,
                )
            return JSONResponse({"error": str(exc)}, status_code=500)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
            if audio_path != tmp.name:
                Path(audio_path).unlink(missing_ok=True)

    # Side-Step only posts to /caption for local URLs, but accept the
    # generic-endpoint names too so a differently-shaped URL still works.
    app.add_api_route("/infer", caption, methods=["POST"])
    app.add_api_route("/analyze", caption, methods=["POST"])

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="path to the downloaded model directory")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (keep loopback unless you know why not)")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"],
                        help="'auto' uses the checkpoint's own dtype")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    # Accept either a local directory or a Hub repo id.
    model_ref = str(Path(args.model).expanduser()) if Path(args.model).expanduser().exists() else args.model
    if "/" not in model_ref and not Path(model_ref).exists():
        print(f"[FAIL] No such model path and not a Hub repo id: {args.model}")
        return 1

    load_model(model_ref, args.dtype)

    import uvicorn
    app = build_app(args.max_new_tokens)
    print(f"[OK] Set Side-Step's Music Flamingo URL to: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
