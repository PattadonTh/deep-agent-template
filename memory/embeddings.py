import os
from dotenv import load_dotenv

load_dotenv()

# ✏️  TODO: Change EMBEDDING_MODEL in .env to use a different model.
#      Larger models are more accurate but slower and use more RAM.
#      Popular alternatives:
#        all-mpnet-base-v2        — better quality, ~420MB
#        paraphrase-MiniLM-L3-v2  — faster, smaller, ~60MB
#        BAAI/bge-small-en-v1.5   — strong multilingual support
#      The model is downloaded automatically on first run from HuggingFace.
#      IMPORTANT: Keep it consistent across runs — changing models means old
#      embeddings are no longer comparable; call store.clear() before switching.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Module-level singleton so the model is loaded once and reused for all calls.
# Loading a SentenceTransformer takes ~1–2 seconds, so lazy-loading here
# avoids that cost on import and only pays it when embeddings are first needed.
_model = None


def _get_model():
    """
    Lazy-load the SentenceTransformer model on first use.

    The global singleton pattern ensures the model weights are only read
    from disk once per process, regardless of how many embed() calls follow.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed(text: str) -> list[float]:
    """
    Embed a single text string into a dense float vector.

    Used at query-time (one topic string → one vector for ANN search).
    The "empty" fallback prevents a blank string from producing a degenerate
    zero-vector that would distort cosine similarity scores.
    """
    if not text.strip():
        text = "empty"
    return _get_model().encode(text).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts in a single forward pass.

    Much faster than calling embed() in a loop because the model processes
    all inputs as a matrix in one GPU/CPU operation.
    Used at save-time when we write a full set of chunks at once.
    """
    texts = [t if t.strip() else "empty" for t in texts]
    return _get_model().encode(texts).tolist()
