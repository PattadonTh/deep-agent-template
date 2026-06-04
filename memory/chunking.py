import uuid


def make_id(topic: str, chunk_type: str, suffix: str = "") -> str:
    """Generate a deterministic unique ID for a chunk."""
    base = f"{topic}::{chunk_type}::{suffix}".lower().replace(" ", "_")
    return base[:80] + "::" + str(uuid.uuid5(uuid.NAMESPACE_DNS, base))


def chunk_text(
    text: str,
    topic: str,
    chunk_type: str = "general",
    max_length: int = 1500,
) -> list[dict]:
    """
    Split plain text into semantic chunks for vector storage.
    Override this per project for structured output schemas.
    """
    chunks = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    current = ""
    index = 0

    for para in paragraphs:
        if len(current) + len(para) > max_length and current:
            chunks.append(
                {
                    "id": make_id(topic, chunk_type, str(index)),
                    "text": f"Topic: {topic}\n{current.strip()}",
                    "metadata": {
                        "topic": topic,
                        "chunk_type": chunk_type,
                        "index": index,
                    },
                }
            )
            current = para
            index += 1
        else:
            current += "\n\n" + para if current else para

    if current:
        chunks.append(
            {
                "id": make_id(topic, chunk_type, str(index)),
                "text": f"Topic: {topic}\n{current.strip()}",
                "metadata": {
                    "topic": topic,
                    "chunk_type": chunk_type,
                    "index": index,
                },
            }
        )

    return chunks
