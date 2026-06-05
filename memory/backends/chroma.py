import os
import chromadb
from ..store import BaseMemoryStore
from ..embeddings import embed, embed_batch
from ..chunking import chunk_text

# Directory where ChromaDB persists data between runs.
# ✏️  TODO: Change CHROMA_DIR in .env if you want to store the DB elsewhere
#      (e.g. a shared Docker volume, or an absolute path on a remote machine).
#      The folder is created automatically on first run.
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")

# All agent memory lives in this single named collection.
# ✏️  TODO: Change COLLECTION_NAME to get isolated memory per project
#      without deleting old data — just point each project to its own name.
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "agent_memory")

# Cosine similarity threshold: chunks below this score are silently dropped.
# Range: 0.0 (accept everything) → 1.0 (exact match only).
# ✏️  TODO: Lower to ~0.1 to get more (noisier) results.
#      Raise to ~0.5+ for strict relevance filtering.
SIMILARITY_THRESHOLD = 0.3


class ChromaMemoryStore(BaseMemoryStore):
    """
    Persistent vector memory backed by ChromaDB (local, file-based).

    ChromaDB stores embeddings on disk (CHROMA_DIR) so memory survives
    between runs without needing an external database service.
    It uses an HNSW cosine index for fast approximate nearest-neighbour
    search across thousands of chunks.

    Lifecycle:
      __init__  — connect to (or create) the on-disk collection
      save()    — chunk → embed → upsert (idempotent: same ID = overwrite)
      query()   — embed the query topic → ANN search → filter by threshold
      format_context() — render results into a prompt-ready string
      clear()   — delete and recreate the collection (destructive!)
    """

    def __init__(self):
        # PersistentClient flushes writes to disk automatically after each call.
        # No explicit commit/flush needed.
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            # hnsw:space=cosine: distance = 1 − cosine_similarity
            # so distance=0.0 is a perfect match, distance=1.0 is orthogonal.
            metadata={"hnsw:space": "cosine"},
        )

    def save(self, text: str, topic: str, chunk_type: str = "general") -> int:
        """
        Chunk, embed, and upsert text into the collection.

        upsert (not insert) means: if a chunk with the same deterministic ID
        already exists it is overwritten, not duplicated.  Safe to call
        multiple times with the same content — the store stays clean.

        Returns the number of chunks written (useful for logging).
        """
        chunks = chunk_text(text, topic=topic, chunk_type=chunk_type)
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        self.collection.upsert(
            ids=[c["id"] for c in chunks],
            embeddings=embed_batch(texts),  # batch embed is faster than one-by-one
            documents=texts,
            metadatas=[c["metadata"] for c in chunks],
        )
        print(f"💾 [Memory] Saved {len(chunks)} chunks for '{topic}'")
        return len(chunks)

    def query(self, topic: str, top_k: int = 5) -> list[dict]:
        """
        Semantic search: find chunks most similar to `topic`.

        Steps:
          1. Guard against querying an empty collection (ChromaDB raises an
             error if n_results > total count).
          2. Embed the topic string into a query vector.
          3. Run ANN search for the nearest top_k chunks.
          4. Convert distances → similarities (similarity = 1 − distance).
          5. Filter out chunks below SIMILARITY_THRESHOLD to avoid injecting
             irrelevant context into the agent's prompt.

        Returns a list of dicts (may be empty if nothing passes the threshold).
        """
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_embeddings=[embed(topic)],
            # min() prevents requesting more results than exist in the collection.
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        memories = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB returns cosine *distance* — convert to similarity.
            similarity = round(1 - dist, 3)
            if similarity < SIMILARITY_THRESHOLD:
                continue  # skip low-relevance chunks
            memories.append({"text": doc, "metadata": meta, "similarity": similarity})
        return memories

    def format_context(self, memories: list[dict]) -> str:
        """
        Render retrieved memory chunks into a prompt-ready string block.

        The output is numbered and tagged with topic/type/relevance so the
        LLM can distinguish individual memory entries.  The "## Past Context"
        header signals to the agent that this section is retrieved memory,
        not part of the current task instructions.

        Returns "" (empty string) when there are no memories so callers
        can use a simple `if context:` check without extra None handling.
        """
        if not memories:
            return ""
        lines = ["## Past Context (retrieved by relevance)\n"]
        for i, mem in enumerate(memories, 1):
            meta = mem["metadata"]
            lines.append(
                f"[{i}] topic={meta.get('topic')} "
                f"type={meta.get('chunk_type')} "
                f"relevance={mem['similarity']:.0%}"
            )
            lines.append(mem["text"])
            lines.append("")
        return "\n".join(lines)

    def count(self) -> int:
        """Return total chunk count in the collection (useful for debugging)."""
        return self.collection.count()

    def clear(self) -> None:
        """
        Delete and recreate the collection, removing all stored memories.

        WARNING: permanently deletes all data.  The collection is immediately
        recreated empty so the store is usable again right after this call.
        Use this when resetting between unrelated projects or during testing.
        """
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("🗑️  [Memory] Cleared.")
