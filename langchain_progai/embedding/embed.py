import requests

from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

from langchain_core.embeddings import Embeddings

from typing import Any, Dict, List, Optional
from langchain_progai.config import get_endpoint


class TextEmbeddingsInference(BaseModel, Embeddings):
    """From text-embeddings-inference api."""

    endpoint: str = get_endpoint("EMBEDDING")
    batch_size: int = 10
    session: Any
    api_key: Optional[SecretStr] = None

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that auth token exists in environment."""
        api_key = convert_to_secret_str(get_from_dict_or_env(values, "api_key", "PROGAI_TOKEN"))
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {api_key.get_secret_value()}",
                "Accept-Encoding": "identity",
                "Content-type": "application/json",
            }
        )
        values["session"] = session
        return values

    def _embed_from_api(self, texts: List[str]) -> List[List[float]]:
        """Internal method to call API and return embeddings.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of list of floats representing the embeddings, or None if an
            error occurs.
        """
        response = self.session.post(self.endpoint, json={"inputs": texts})
        if response.status_code == 200:
            return response.json()
        else:
            raise requests.HTTPError()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of documents.

        Args:
            texts: The list of texts to embed.

        Returns:
            A list of embeddings, one for each text.
        """
        all_embeddings = []

        # Split the texts into chunks of size self.batch_size
        chunks = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        for chunk in chunks:
            embeddings = self._embed_from_api(chunk)
            if embeddings is not None:
                all_embeddings.extend(embeddings)
            else:
                raise ConnectionError()

        return all_embeddings if len(all_embeddings) > 0 else None

    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a single query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text, or None if an error occurs.
        """
        result = self._embed_from_api([text])
        if result is None or len(result) < 1:
            raise ConnectionError()
        else:
            return result[0]
