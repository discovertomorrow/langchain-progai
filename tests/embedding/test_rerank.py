import pytest
from langchain_core.documents import Document

import langchain_progai.embedding as embedding


class TestRerankCompressor:

    @pytest.mark.integration
    def test_compress_documents___given_simple_example___acts_as_expected(self):
        # Arrange
        top_k = 2
        reranker = embedding.RerankCompressor(top_k=top_k)
        documents = [
            Document(page_content="Einstein"),
            Document(page_content="Gandhi"),
            Document(page_content="Newton"),
        ]
        query = "Famous Physicists"
        # Act
        compressed = reranker.compress_documents(documents=documents, query=query)

        # Assert
        assert len(compressed) == top_k
        assert "Gandhi" not in [d.page_content for d in compressed]
        assert all([d in documents for d in compressed])

    @pytest.mark.integration
    def test_compress_documents___given_wrong_progai_token___raises_exception(cls, monkeypatch):
        # Arrange
        monkeypatch.setenv("PROGAI_TOKEN", "invalid_token")
        reranker = embedding.RerankCompressor(top_k=1)
        documents = [
            Document(page_content="Einstein"),
            Document(page_content="Gandhi"),
            Document(page_content="Newton"),
        ]
        query = "Famous Physicists"

        # Act & Assert
        with pytest.raises(PermissionError):
            reranker.compress_documents(documents=documents, query=query)
