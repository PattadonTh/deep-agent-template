import os


def get_memory_store():
    backend = os.getenv("MEMORY_BACKEND", "chroma")
    if backend == "chroma":
        from .chroma import ChromaMemoryStore

        return ChromaMemoryStore()
    raise ValueError(f"Unknown backend: {backend}")
