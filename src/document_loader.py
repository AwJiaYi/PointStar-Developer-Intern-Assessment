from pathlib import Path


def load_document(path: str | Path) -> str:
    """Load the single sample document used to ground the agent."""
    document_path = Path(path)

    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    text = document_path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Document is empty: {document_path}")

    return text
