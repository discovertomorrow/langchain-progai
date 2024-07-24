import numpy as np
import pytest

import langchain_progai.embedding as embedding


class TestTextEmbeddingsInference:

    @pytest.mark.integration
    def test_embed_documents___given_simple_input___returns_valid_embedding(self):
        # Arrange
        texts = ["1", "2", "3"]

        # Act
        embedded = embedding.TextEmbeddingsInference().embed_documents(texts)

        # Assert
        assert len(embedded) == len(texts)
        assert len(embedded[0]) > 0
        assert all([len(vec) == len(embedded[0]) for vec in embedded])

    @pytest.mark.integration
    def test_embed_documents___run_twice___yields_same_results(self):
        # Arrange & Act
        embedded_1 = embedding.TextEmbeddingsInference().embed_query("test query")
        embedded_2 = embedding.TextEmbeddingsInference().embed_query("test query")

        # Assert
        assert np.allclose(embedded_1, embedded_2)

    @pytest.mark.integration
    def test_invoke___given_wrong_progai_token___raises_exception(cls, monkeypatch):
        # Arrange
        monkeypatch.setenv("PROGAI_TOKEN", "invalid_token")
        model = embedding.TextEmbeddingsInference()

        # Act & Assert
        with pytest.raises(PermissionError):
            model.embed_query("Who are you?")
