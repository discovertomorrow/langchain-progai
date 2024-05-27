# This file includes code adapted from the langchain_community project, which
# is also licensed under the MIT License.
# The original code can be found at:
# https://github.com/langchain-ai/langchain/blob/v0.1.0/libs/community/langchain_community/llms/ollama.py
# Changes have been made to the original code to fit the needs of this project.

# Original LICENSE Text:
# MIT License

# Copyright (c) LangChain, Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
import re
import time

import requests
import aiohttp

from langchain_core.pydantic_v1 import SecretStr, validator, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_env
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseLLM, BaseLanguageModel
from langchain_core.outputs import GenerationChunk, LLMResult

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_progai.config import get_endpoint


def _stream_response_to_generation_chunk(
    stream_response: str,
) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""
    parsed_response = json.loads(stream_response)
    generation_info = parsed_response if parsed_response.get("stop") is True else None
    return GenerationChunk(text=parsed_response.get("content", ""), generation_info=generation_info)


def _make_request_with_rate_limit_handling(request_function, *args, **kwargs):
    while True:
        response = request_function(*args, **kwargs)
        if response.status_code == 429:
            # Rate Limit erreicht
            retry_after = int(response.headers.get("Retry-After", 10))
            print(f"Rate limit reached, waiting for {retry_after} seconds.")
            time.sleep(retry_after)
            continue
        break
    return response


class _LlamaCppServerCommon(BaseLanguageModel):
    base_url: str | None = None
    """Base url the model is hosted under."""

    mirostat: Optional[int] = 0
    """Enable Mirostat sampling for controlling perplexity.
    (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)"""

    mirostat_eta: Optional[float] = 0.1
    """Influences how quickly the algorithm responds to feedback
    from the generated text. A lower learning rate will result in
    slower adjustments, while a higher learning rate will make
    the algorithm more responsive. (Default: 0.1)"""

    mirostat_tau: Optional[float] = 5.0
    """Controls the balance between coherence and diversity
    of the output. A lower value will result in more focused and
    coherent text. (Default: 5.0)"""

    repeat_last_n: Optional[int] = 64
    """Sets how far back for the model to look back to prevent
    repetition. (Default: 64, 0 = disabled, -1 = num_ctx)"""

    repeat_penalty: Optional[float] = 1.1
    """Sets how strongly to penalize repetitions. A higher value (e.g., 1.5)
    will penalize repetitions more strongly, while a lower value (e.g., 0.9)
    will be more lenient. (Default: 1.1)"""

    temperature: Optional[float] = 0.8
    """The temperature of the model. Increasing the temperature will
    make the model answer more creatively. (Default: 0.8)"""

    stop: Optional[List[str]] = ["</s>", "<s>", "<|system|>", "<|user|>", "<|assistant|>"]
    """Sets the stop tokens to use."""

    tfs_z: Optional[float] = 1.0
    """Tail free sampling is used to reduce the impact of less probable
    tokens from the output. A higher value (e.g., 2.0) will reduce the
    impact more, while a value of 1.0 disables this setting. (default: 1)"""

    top_k: Optional[int] = 40
    """Reduces the probability of generating nonsense. A higher value (e.g. 100)
    will give more diverse answers, while a lower value (e.g. 10)
    will be more conservative. (Default: 40)"""

    top_p: Optional[int] = 0.9
    """Works together with top-k. A higher value (e.g., 0.95) will lead
    to more diverse text, while a lower value (e.g., 0.5) will
    generate more focused and conservative text. (Default: 0.9)"""

    tokens_to_fix: Optional[List[str]] = ["<s>", "</s>"]
    cache_prompt: bool = True
    n_predict: int = 2500
    slot_id: int = -1

    seed: int = -1

    api_key: Optional[SecretStr] = None

    @root_validator(allow_reuse=True)
    def validate_endpoint(cls, values: Dict) -> Dict:
        """If no base_url is set explicitly, try to fallback to configuration file or environment variables."""
        values["base_url"] = values["base_url"] or get_endpoint("ZEPHYR7B")
        return values

    @validator("api_key", always=True)
    def api_key_must_exist(cls, v: Optional[SecretStr]) -> SecretStr:
        """Validate that auth token exists (in environment)."""
        if v is not None:
            return v
        return convert_to_secret_str(get_from_env("api_key", "PROGAI_TOKEN"))

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling LlamaCpp server."""
        return {
            "mirostat": self.mirostat,
            "mirostat_eta": self.mirostat_eta,
            "mirostat_tau": self.mirostat_tau,
            "repeat_last_n": self.repeat_last_n,
            "repeat_penalty": self.repeat_penalty,
            "temperature": self.temperature,
            "stop": self.stop,
            "tfs_z": self.tfs_z,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "seed": self.seed,
            "n_predict": self.n_predict,
            "slot_id": self.slot_id,
        }

    def _create_stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        if stop is None:
            stop = self.stop

        params = self._default_params
        params["stop"] = stop
        if "image_data" in kwargs:
            params["image_data"] = kwargs["image_data"]

        prompt = fix_prompt_for_llamacpp_tokenization(prompt, self.tokens_to_fix) if self.tokens_to_fix else prompt

        response = _make_request_with_rate_limit_handling(
            requests.post,
            url=f"{self.base_url}",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            },
            json={"prompt": prompt, "stream": True, "cache_prompt": self.cache_prompt, **params},
            stream=True,
        )

        response.encoding = "utf-8"
        if response.status_code != 200:
            raise ValueError(f"LlamaCppServer call failed with status code {response.status_code}.")
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                yield line.replace("data: ", "", 1)

    async def _acreate_stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        if stop is None:
            stop = self.stop

        params = self._default_params
        params["stop"] = stop
        if "image_data" in kwargs:
            params["image_data"] = kwargs["image_data"]

        prompt = fix_prompt_for_llamacpp_tokenization(prompt, self.tokens_to_fix) if self.tokens_to_fix else prompt

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=f"{self.base_url}",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key.get_secret_value()}",
                },
                json={"prompt": prompt, "stream": True, "cache_prompt": self.cache_prompt, **params},
            ) as response:
                if response.status != 200:
                    raise ValueError(f"LlamaCppServer call failed with status code {response.status}.")
                async for line in response.content:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        yield decoded.replace("data: ", "", 1)

    def _stream_with_aggregation(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> GenerationChunk:
        final_chunk: Optional[GenerationChunk] = None
        for stream_resp in self._create_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_generation_chunk(stream_resp)
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from LlamaCppServer stream.")

        return final_chunk


class LlamaCppServer(_LlamaCppServerCommon, BaseLLM):
    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = False

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "llamacpp-llm"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            final_chunk = super()._stream_with_aggregation(
                prompt,
                stop=stop,
                run_manager=run_manager,
                verbose=self.verbose,
                **kwargs,
            )
            generations.append([final_chunk])
        return LLMResult(generations=generations)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for stream_resp in self._create_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_generation_chunk(stream_resp)
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        async for stream_resp in self._acreate_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_generation_chunk(stream_resp)
                yield chunk
                if run_manager:
                    await run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )


def tokenize(base_url, content: str) -> List[int]:
    response = requests.post(
        url=f"{base_url}/tokenize",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('PROGAI_TOKEN', '')}"},
        json={"content": content},
    )
    return response.json()["tokens"]


def fix_prompt_for_llamacpp_tokenization(s, tokens_to_fix):
    for token in tokens_to_fix:
        escaped_token = re.escape(token)
        regex_pattern = rf"{escaped_token}(?!$)"
        s = re.sub(regex_pattern, f"{token} ", s)
    return s
