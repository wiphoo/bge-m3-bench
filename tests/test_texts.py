"""Tests for the shared input-text loader (``common.texts.load_texts``)."""

from __future__ import annotations

import urllib.error

import pytest

from bge_m3_bench.common import texts as texts_mod
from bge_m3_bench.common.texts import DEFAULT_TEXTS, load_texts


def test_load_texts_default_when_none() -> None:
    assert load_texts(None) == list(DEFAULT_TEXTS)
    assert load_texts("") == list(DEFAULT_TEXTS)


def test_load_texts_local_file(tmp_path) -> None:
    f = tmp_path / "inputs.txt"
    f.write_text("alpha\n  beta  \n\ngamma\n")
    # Blank lines dropped, surrounding whitespace stripped.
    assert load_texts(str(f)) == ["alpha", "beta", "gamma"]


def test_load_texts_missing_local_file_raises_value_error(tmp_path) -> None:
    missing = tmp_path / "nope.txt"
    with pytest.raises(ValueError, match="failed to read texts from"):
        load_texts(str(missing))


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_load_texts_url_fetch(monkeypatch, scheme: str) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["user_agent"] = req.get_header("User-agent")
        return _FakeResponse(b"one\n  two  \n\nthree\n")

    monkeypatch.setattr(texts_mod.urllib.request, "urlopen", fake_urlopen)

    url = f"{scheme}://example.invalid/inputs.txt"
    assert load_texts(url) == ["one", "two", "three"]
    assert captured["url"] == url
    assert captured["timeout"] == 30
    # A User-Agent must be sent — the CDN 403s requests without one.
    assert captured["user_agent"]


def test_load_texts_url_failure_raises_value_error(monkeypatch) -> None:
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(texts_mod.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="failed to fetch texts from"):
        load_texts("https://example.invalid/inputs.txt")


def test_load_texts_url_non_utf8_raises_value_error(monkeypatch) -> None:
    def fake_urlopen(req, timeout=None):
        return _FakeResponse(b"\xff\xfe not utf-8")

    monkeypatch.setattr(texts_mod.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="failed to fetch texts from"):
        load_texts("https://example.invalid/inputs.txt")


class _ReadTimeoutResponse(_FakeResponse):
    """Connects fine but stalls/raises while reading the response body."""

    def __init__(self) -> None:
        super().__init__(b"")

    def read(self) -> bytes:
        raise TimeoutError("timed out")


def test_load_texts_url_read_timeout_raises_value_error(monkeypatch) -> None:
    # A stalled body read raises TimeoutError (not URLError); it must still be
    # wrapped as a clear ValueError rather than escaping as a traceback.
    def fake_urlopen(req, timeout=None):
        return _ReadTimeoutResponse()

    monkeypatch.setattr(texts_mod.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="failed to fetch texts from"):
        load_texts("https://example.invalid/inputs.txt")
