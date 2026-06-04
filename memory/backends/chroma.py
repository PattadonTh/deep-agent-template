import os
import chromadb
from ..store import BaseMemoryStore
from ..embeddings import embed, embed_batch
from ..chunking import chunk_text

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "agent_memory")
SIMILARITY_THRESHOLD = 0.3


class ChromaMemoryStore(BaseMemoryStore):

    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def save(self, text: str, topic: str, chunk_type: str = "general") -> int:
        chunks = chunk_text(text, topic=topic, chunk_type=chunk_type)
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        self.collection.upsert(
            ids=[c["id"] for c in chunks],
            embeddings=embed_batch(texts),
            documents=texts,
            metadatas=[c["metadata"] for c in chunks],
        )
        print(f"💾 [Memory] Saved {len(chunks)} chunks for '{topic}'")
        return len(chunks)

    def query(self, topic: str, top_k: int = 5) -> list[dict]:
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_embeddings=[embed(topic)],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        memories = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = round(1 - dist, 3)
            if similarity < SIMILARITY_THRESHOLD:
                continue
            memories.append({"text": doc, "metadata": meta, "similarity": similarity})
        return memories

    def format_context(self, memories: list[dict]) -> str:
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
        return self.collection.count()

    def clear(self) -> None:
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("🗑️  [Memory] Cleared.")
