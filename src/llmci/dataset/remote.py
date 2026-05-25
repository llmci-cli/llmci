"""Remote dataset resolution and download."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from llmci.errors import DatasetError
from llmci.models import DatasetSource

DEFAULT_CACHE_DIR = Path(".llmci/cache/datasets")


def is_remote_uri(source: str) -> bool:
    """Return True if the source is a remote URI rather than a local path."""
    return source.startswith(("s3://", "https://", "http://"))


def resolve_dataset_path(
    dataset: str | Path | DatasetSource,
    cache_dir: Path | None = None,
) -> Path:
    """Resolve a dataset reference to a local JSONL path.

    Local paths are returned as-is. Remote URIs are downloaded to the cache
    directory (or a temp file when caching is disabled).
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR

    if isinstance(dataset, DatasetSource):
        return fetch_remote_dataset(dataset.source, cache_dir, use_cache=dataset.cache)

    source = str(dataset)
    if is_remote_uri(source):
        return fetch_remote_dataset(source, cache_dir, use_cache=True)

    return Path(dataset)


def fetch_remote_dataset(
    source: str,
    cache_dir: Path,
    *,
    use_cache: bool = True,
) -> Path:
    """Download a remote dataset and return the local path."""
    if not is_remote_uri(source):
        raise DatasetError(f"Not a remote dataset URI: {source}")

    if use_cache:
        dest = _cache_path(source, cache_dir)
        if dest.exists():
            return dest
    else:
        suffix = _filename_from_uri(source)
        tmp = tempfile.NamedTemporaryFile(
            prefix="llmci-dataset-",
            suffix=suffix,
            delete=False,
        )
        tmp.close()
        dest = Path(tmp.name)

    try:
        if source.startswith("s3://"):
            _download_s3(source, dest)
        else:
            _download_http(source, dest)
    except DatasetError:
        if not use_cache and dest.exists():
            dest.unlink(missing_ok=True)
        raise
    except Exception as e:
        if not use_cache and dest.exists():
            dest.unlink(missing_ok=True)
        raise DatasetError(
            f"Failed to download dataset from {source}:\n  {e}\n\n"
            "Fix: Check the URI, credentials, and network access."
        ) from e

    return dest


def _cache_path(source: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    return cache_dir / f"{digest}_{_filename_from_uri(source)}"


def _filename_from_uri(source: str) -> str:
    name = Path(urlparse(source).path).name
    return name or "dataset.jsonl"


def _download_s3(uri: str, dest: Path) -> None:
    try:
        import boto3  # type: ignore[import-not-found,import-untyped]
    except ImportError as e:
        raise DatasetError(
            "S3 datasets require boto3.\n\n"
            "Fix: pip install 'llmci[s3]'"
        ) from e

    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise DatasetError(
            f"Invalid S3 URI: {uri}\n\n"
            "Fix: Use s3://bucket-name/path/to/dataset.jsonl"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, str(dest))


def _download_http(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, dest.open("wb") as out:
        out.write(response.read())
