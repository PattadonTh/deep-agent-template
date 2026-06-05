from abc import ABC, abstractmethod


class BaseMemoryStore(ABC):
    """
    Abstract interface that every memory backend must implement.

    New backends (e.g. Pinecone, Weaviate, Redis) only need to subclass this
    and implement the five methods below.  The rest of the codebase only talks
    to this interface, so switching backends is a one-line .env change.

    Method summary:
      save()           — chunk text, embed it, and persist to the store
      query()          — semantic search: find chunks similar to a topic string
      format_context() — render retrieved chunks into a prompt-ready string
      count()          — how many chunks are currently stored
      clear()          — wipe the entire collection (use with caution)
    """

    @abstractmethod
    def save(self, text: str, topic: str, chunk_type: str = "general") -> int:
        """
        Chunk, embed, and store `text` tagged with `topic`.

        Returns the number of chunks that were written.
        chunk_type is a free-form label (e.g. "general", "code", "summary")
        stored in metadata so you can filter by it later if needed.
        """
        pass

    @abstractmethod
    def query(self, topic: str, top_k: int = 5) -> list[dict]:
        """
        Semantic search: return up to `top_k` chunks most similar to `topic`.

        Each item in the returned list is a dict with keys:
          text       — the chunk text that will be injected into the prompt
          metadata   — topic, chunk_type, index
          similarity — cosine similarity score in [0, 1]; higher = more relevant
        """
        pass

    @abstractmethod
    def format_context(self, memories: list[dict]) -> str:
        """
        Convert the list returned by query() into a formatted string
        ready to be inserted into the agent's prompt.

        Returns an empty string if `memories` is empty so callers can
        safely do `if context:` without extra checks.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the total number of chunks currently in the store."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Delete all stored chunks and recreate an empty collection.

        WARNING: this is irreversible.  Use only for testing or resetting
        between unrelated projects.
        """
        pass
