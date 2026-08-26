import math

import pytest

from tabs.score.embeddings import EMBEDDING_MODEL, cosine_similarity, embed_text


class _FakeEmbeddingsResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _FakeVoyageClient:
    def __init__(self, embedding):
        self._embedding = embedding
        self.calls = []

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeEmbeddingsResult([self._embedding])


def test_embed_text_returns_the_embedding_vector():
    client = _FakeVoyageClient([0.1, 0.2, 0.3])

    result = embed_text(client, "A claim about a vulnerability")

    assert result == [0.1, 0.2, 0.3]


def test_embed_text_uses_the_embedding_model_and_document_input_type():
    client = _FakeVoyageClient([0.1, 0.2, 0.3])

    embed_text(client, "text")

    assert client.calls[0]["model"] == EMBEDDING_MODEL
    assert client.calls[0]["input_type"] == "document"
    assert client.calls[0]["texts"] == ["text"]


def test_cosine_similarity_of_identical_vectors_is_one():
    assert math.isclose(cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-9)


def test_cosine_similarity_of_opposite_vectors_is_negative_one():
    assert math.isclose(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)


def test_cosine_similarity_handles_a_zero_vector_without_dividing_by_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_raises_on_mismatched_vector_lengths():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
