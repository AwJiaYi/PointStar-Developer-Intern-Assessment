from pathlib import Path

import pytest

from src.document_loader import load_document


def test_load_document(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Annual leave is 14 days.", encoding="utf-8")
    assert load_document(file_path) == "Annual leave is 14 days."


def test_missing_document(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "missing.txt")


def test_empty_document(tmp_path: Path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   ", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_document(file_path)
