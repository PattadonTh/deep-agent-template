import uuid


def make_id(topic: str, chunk_type: str, suffix: str = "") -> str:
    """
    Generate a deterministic, collision-resistant ID for a chunk.

    The ID is built from topic + chunk_type + suffix so that re-saving the
    same content produces the same ID.  This is intentional: ChromaDB's
    upsert will overwrite the old embedding instead of creating a duplicate,
    keeping the store clean across repeated runs.

    The uuid5 suffix (DNS namespace) adds enough entropy to prevent accidental
    collisions when different topics produce the same slug after lowercasing.
    """
    base = f"{topic}::{chunk_type}::{suffix}".lower().replace(" ", "_")
    return base[:80] + "::" + str(uuid.uuid5(uuid.NAMESPACE_DNS, base))


def chunk_text(
    text: str,
    topic: str,
    chunk_type: str = "general",
    # ✏️  TODO: Tune max_length to balance retrieval quality vs embedding cost.
    #      Smaller chunks = more precise retrieval but more chunks to store.
    #      Larger chunks = more context per result but noisier matches.
    #      1500 chars ≈ ~300–400 tokens, a good default for most LLMs.
    max_length: int = 1500,
) -> list[dict]:
    """
    Split plain text into paragraph-based chunks for vector storage.

    Strategy:
      1. Split the text on double newlines (paragraph boundaries).
      2. Accumulate paragraphs into a running buffer until adding the next one
         would exceed max_length.
      3. When the buffer is full, flush it as a chunk and start a new one.

    Each chunk dict has the shape expected by the backend's upsert call:
      id       — deterministic ID from make_id() (enables safe upsert/overwrite)
      text     — "Topic: <topic>\\n<content>" (topic prefix helps retrieval quality)
      metadata — topic, chunk_type, index (for filtering / debugging)

    ✏️  TODO: Override this function if your AgentOutput has structured fields
    (e.g. title, summary, code).  Create one chunk per field for precise retrieval:
      return [
          {"id": make_id(topic, "title", "0"), "text": f"Title: {result.title}", ...},
          {"id": make_id(topic, "summary", "0"), "text": result.summary, ...},
      ]
    """
    chunks = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    current = ""
    index = 0

    for para in paragraphs:
        if len(current) + len(para) > max_length and current:
            # Buffer is full — flush as a chunk before starting a new one.
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

    # Flush whatever remains in the buffer as the final chunk.
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
