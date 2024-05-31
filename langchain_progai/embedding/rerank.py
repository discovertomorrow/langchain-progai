import requests
import os
import asyncio

from typing import Optional, Sequence, Dict

from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.pydantic_v1 import root_validator

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_progai.config import get_endpoint


class RerankCompressor(BaseDocumentCompressor):
    endpoint: str | None = None
    top_k: int = 3

    headers = {"Content-Type": "application/json"}
    token = os.getenv("PROGAI_TOKEN", "")

    @root_validator(allow_reuse=True)
    def validate_endpoint(cls, values: Dict) -> Dict:
        """If no endpoint is set explicitly, try to fallback to configuration file or environment variables."""
        values["endpoint"] = values["endpoint"] or get_endpoint(
            "RERANKER_BGE_L")
        return values

    def _generate_headers(self):
        headers = self.headers
        if self.token != "":
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get_scores_from_endpoint(
        self,
        documents: Sequence[Document],
        query: str,
    ):
        data = {"query": query, "texts": [
            document.page_content for document in documents]}
        response = requests.post(
            self.endpoint, headers=self._generate_headers(), json=data)

        if response.status_code == 200:
            return [(s["index"], s["score"]) for s in response.json()]
        else:
            print("Error:", response.status_code, response.text)
            return None

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        scores = self._get_scores_from_endpoint(documents, query)
        top_k_scores = sorted(scores, key=lambda x: x[1], reverse=True)[
            :self.top_k]
        result = [(score, documents[index]) for index, score in top_k_scores]
        return [doc for _, doc in result]

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self.compress_documents, documents, query, callbacks
        )
