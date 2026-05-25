"""Tests for remote dataset resolution."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llmci.dataset.loader import load_dataset
from llmci.dataset.remote import fetch_remote_dataset, is_remote_uri, resolve_dataset_path
from llmci.errors import DatasetError
from llmci.models import DatasetSource


class TestIsRemoteUri:
    def test_s3(self):
        assert is_remote_uri("s3://bucket/path/data.jsonl")

    def test_https(self):
        assert is_remote_uri("https://example.com/data.jsonl")

    def test_local_path(self):
        assert not is_remote_uri("./evals/tickets.jsonl")


class TestResolveDatasetPath:
    def test_local_path(self, tmp_path):
        dataset = tmp_path / "data.jsonl"
        dataset.write_text('{"input": "a", "expected": "b"}\n')
        assert resolve_dataset_path(str(dataset), cache_dir=tmp_path / "cache") == dataset

    def test_remote_string_uses_cache(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cached = cache_dir / "abc_tickets.jsonl"
        cached.parent.mkdir(parents=True)
        cached.write_text('{"input": "cached", "expected": "ok"}\n')

        with patch(
            "llmci.dataset.remote.fetch_remote_dataset",
            return_value=cached,
        ) as fetch:
            result = resolve_dataset_path(
                "s3://my-bucket/tickets.jsonl",
                cache_dir=cache_dir,
            )

        assert result == cached
        fetch.assert_called_once_with(
            "s3://my-bucket/tickets.jsonl",
            cache_dir,
            use_cache=True,
        )

    def test_dataset_source_respects_cache_flag(self, tmp_path):
        cache_dir = tmp_path / "cache"
        source = DatasetSource(source="https://example.com/data.jsonl", cache=False)

        with patch("llmci.dataset.remote.fetch_remote_dataset") as fetch:
            fetch.return_value = tmp_path / "downloaded.jsonl"
            resolve_dataset_path(source, cache_dir=cache_dir)

        fetch.assert_called_once_with(
            "https://example.com/data.jsonl",
            cache_dir,
            use_cache=False,
        )


class TestFetchRemoteDataset:
    def test_uses_cache_when_present(self, tmp_path):
        cache_dir = tmp_path / "cache"
        source = "https://example.com/tickets.jsonl"

        def fake_download(_url: str, dest: Path) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("{}\n")

        with patch("llmci.dataset.remote._download_http", side_effect=fake_download) as download:
            first = fetch_remote_dataset(source, cache_dir, use_cache=True)
            second = fetch_remote_dataset(source, cache_dir, use_cache=True)

        assert first == second
        download.assert_called_once()

    def test_http_download(self, tmp_path):
        cache_dir = tmp_path / "cache"
        payload = b'{"input": "remote", "expected": "value"}\n'

        with patch("llmci.dataset.remote.urlopen") as urlopen:
            response = MagicMock()
            response.read.return_value = payload
            response.__enter__.return_value = response
            response.__exit__.return_value = False
            urlopen.return_value = response

            path = fetch_remote_dataset(
                "https://example.com/tickets.jsonl",
                cache_dir,
                use_cache=True,
            )

        assert path.exists()
        assert path.read_bytes() == payload

    def test_s3_requires_boto3(self, tmp_path):
        from llmci.dataset.remote import _download_s3

        dest = tmp_path / "out.jsonl"

        def _import_error(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("no boto3")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import_error):
            with pytest.raises(DatasetError, match="boto3"):
                _download_s3("s3://bucket/tickets.jsonl", dest)

    def test_s3_download(self, tmp_path):
        cache_dir = tmp_path / "cache"
        mock_client = MagicMock()

        with patch("boto3.client", return_value=mock_client) as client:
            path = fetch_remote_dataset(
                "s3://my-bucket/path/tickets.jsonl",
                cache_dir,
                use_cache=True,
            )

        client.assert_called_once_with("s3")
        mock_client.download_file.assert_called_once_with(
            "my-bucket",
            "path/tickets.jsonl",
            str(path),
        )
        assert path.name.endswith("tickets.jsonl")

    def test_invalid_s3_uri(self, tmp_path):
        with pytest.raises(DatasetError, match="Invalid S3 URI"):
            fetch_remote_dataset("s3://bucket-only", tmp_path / "cache")


class TestLoadDatasetRemote:
    def test_loads_from_http_dataset(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        payload = b'{"input": "remote", "expected": "value"}\n'

        with patch("llmci.dataset.remote.urlopen") as urlopen:
            response = MagicMock()
            response.read.return_value = payload
            response.__enter__.return_value = response
            response.__exit__.return_value = False
            urlopen.return_value = response

            examples = load_dataset("https://example.com/tickets.jsonl")

        assert len(examples) == 1
        assert examples[0].input == "remote"
