import math

EMBEDDING_MODEL = "voyage-4-lite"


def embed_text(voyage_client, text: str) -> list[float]:
    """Embed a single claim's text via Voyage AI, for corroboration/conflict matching.

    input_type="document" is used consistently for every embedding this project makes:
    Voyage's query/document distinction is tuned for asymmetric retrieval (a short query
    against long documents), which doesn't describe this project's symmetric claim-vs-claim
    comparison — treating every claim as a "document" keeps both sides of every comparison
    embedded the same way, which is what a fair similarity comparison requires.
    """
    result = voyage_client.embed(texts=[text], model=EMBEDDING_MODEL, input_type="document")
    return result.embeddings[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length embedding vectors, in [-1.0, 1.0]."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
