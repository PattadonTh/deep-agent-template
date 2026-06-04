import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed(text: str) -> list[float]:
    """Embed a single text string using a local sentence-transformers model."""
    if not text.strip():
        text = "empty"
    return _get_model().encode(text).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in one batch pass."""
    texts = [t if t.strip() else "empty" for t in texts]
    return _get_model().encode(texts).tolist()
