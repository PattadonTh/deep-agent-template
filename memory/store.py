from abc import ABC, abstractmethod


class BaseMemoryStore(ABC):

    @abstractmethod
    def save(self, text: str, topic: str, chunk_type: str = "general") -> int:
        pass

    @abstractmethod
    def query(self, topic: str, top_k: int = 5) -> list[dict]:
        pass

    @abstractmethod
    def format_context(self, memories: list[dict]) -> str:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass
