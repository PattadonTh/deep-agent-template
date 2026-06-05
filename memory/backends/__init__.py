import os


def get_memory_store():
    """
    Factory function — return the correct memory backend based on .env config.

    Set MEMORY_BACKEND in your .env to choose a backend:
      MEMORY_BACKEND=chroma  (default) — local ChromaDB, no external service needed

    ✏️  TODO: To add a new backend (e.g. Pinecone, Redis):
      1. Create memory/backends/pinecone.py with a class extending BaseMemoryStore.
      2. Add an elif branch below:
           elif backend == "pinecone":
               from .pinecone import PineconeMemoryStore
               return PineconeMemoryStore()
      3. Set MEMORY_BACKEND=pinecone in .env — no other code changes needed.

    Imports are intentionally inside the if/elif so unused backend
    dependencies don't need to be installed.
    """
    backend = os.getenv("MEMORY_BACKEND", "chroma")
    if backend == "chroma":
        from .chroma import ChromaMemoryStore

        return ChromaMemoryStore()
    raise ValueError(f"Unknown backend: {backend}")
