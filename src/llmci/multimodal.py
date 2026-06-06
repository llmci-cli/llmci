"""Multimodal message building for direct (litellm) targets.

Dataset rows can attach media beside the text ``input`` using ``images`` and/or
``audio`` fields (stored on ``EvalExample.extra``). Each value is an HTTPS URL or a
path relative to the dataset file::

    {"input": "What is shown?", "expected": "a cat", "images": ["fixtures/cat.png"]}
    {"input": "Transcribe this clip", "expected": "hello", "audio": ["clips/hello.wav"]}

Local files are inlined as data URLs so caching keys stay stable across machines.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from llmci.models import EvalExample

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mpeg", ".mpga"}


def dataset_media_base(dataset: str | Path | object) -> Path | None:
    """Return the directory used to resolve relative media paths, if known."""
    if isinstance(dataset, Path):
        return dataset.parent if dataset.is_absolute() else (Path.cwd() / dataset).parent
    if isinstance(dataset, str):
        path = Path(dataset)
        return path.parent if path.is_absolute() else (Path.cwd() / path).parent
    return None


def has_media(example: EvalExample) -> bool:
    return bool(_media_lists(example)[0] or _media_lists(example)[1])


def media_cache_params(example: EvalExample) -> dict[str, list[str]]:
    """Stable media references for response-cache keys."""
    images, audio = _media_lists(example)
    params: dict[str, list[str]] = {}
    if images:
        params["images"] = images
    if audio:
        params["audio"] = audio
    return params


def build_user_content(
    prompt: str,
    example: EvalExample,
    media_base: Path | None,
) -> str | list[dict]:
    """Build litellm ``messages[].content`` for text-only or multimodal input."""
    images, audio = _media_lists(example)
    if not images and not audio:
        return prompt

    parts: list[dict] = [{"type": "text", "text": prompt}]
    for ref in images:
        url = _resolve_media_ref(str(ref), media_base, kind="image")
        parts.append({"type": "image_url", "image_url": {"url": url}})
    for ref in audio:
        data, fmt = _resolve_audio_payload(str(ref), media_base)
        parts.append({"type": "input_audio", "input_audio": {"data": data, "format": fmt}})
    return parts


def _media_lists(example: EvalExample) -> tuple[list[str], list[str]]:
    return (_as_str_list(example.extra.get("images")), _as_str_list(example.extra.get("audio")))


def _as_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)]


def _is_remote(ref: str) -> bool:
    return ref.startswith(("http://", "https://", "data:"))


def _resolve_media_ref(ref: str, media_base: Path | None, *, kind: str) -> str:
    if _is_remote(ref):
        return ref
    path = Path(ref)
    if not path.is_absolute():
        if media_base is None:
            raise ValueError(
                f"Relative {kind} path {ref!r} requires a dataset file path to resolve against"
            )
        path = media_base / path
    if not path.exists():
        raise ValueError(f"{kind.capitalize()} file not found: {path}")
    if kind == "image":
        ext = path.suffix.lower()
        if ext and ext not in _IMAGE_EXTS:
            raise ValueError(
                f"Unsupported image extension {ext!r} on {path.name}; "
                f"supported: {', '.join(sorted(_IMAGE_EXTS))}"
            )
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _resolve_audio_payload(ref: str, media_base: Path | None) -> tuple[str, str]:
    if _is_remote(ref):
        if ref.startswith("data:"):
            header, encoded = ref.split(",", 1)
            fmt = _audio_format_from_mime(header.split(";")[0].removeprefix("data:"))
            return encoded, fmt
        raise ValueError(
            f"Remote audio URL {ref!r} is not supported; use a local file path or data: URL"
        )

    path = Path(ref)
    if not path.is_absolute():
        if media_base is None:
            raise ValueError(
                f"Relative audio path {ref!r} requires a dataset file path to resolve against"
            )
        path = media_base / path
    if not path.exists():
        raise ValueError(f"Audio file not found: {path}")

    ext = path.suffix.lower()
    if ext not in _AUDIO_EXTS:
        raise ValueError(
            f"Unsupported audio extension {ext!r} on {path.name}; "
            f"supported: {', '.join(sorted(_AUDIO_EXTS))}"
        )
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return encoded, _audio_format_from_ext(ext)


def _audio_format_from_ext(ext: str) -> str:
    mapping = {
        ".wav": "wav",
        ".mp3": "mp3",
        ".mpeg": "mp3",
        ".mpga": "mp3",
        ".m4a": "m4a",
        ".ogg": "ogg",
        ".flac": "flac",
    }
    return mapping.get(ext, ext.lstrip("."))


def _audio_format_from_mime(mime: str) -> str:
    mapping = {
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/mp4": "m4a",
        "audio/ogg": "ogg",
        "audio/flac": "flac",
    }
    return mapping.get(mime, "wav")
