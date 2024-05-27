# This file includes code adapted from the langchain_community project, which
# is also licensed under the MIT License.
# The original code can be found at:
# https://github.com/langchain-ai/langchain/blob/v0.1.0/libs/community/langchain_community/chat_models/ollama.py
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
import jinja2
from typing import Any, AsyncIterator, Iterator, List, Optional, Dict

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import root_validator

from langchain_progai.completion.llamacpp import _LlamaCppServerCommon
from .chat_model import ProgaiChatModel
from langchain_progai.config import get_endpoint


def _stream_response_to_chat_generation_chunk(
    stream_response: str,
) -> ChatGenerationChunk:
    """Convert a stream response to a generation chunk."""
    parsed_response = json.loads(stream_response)
    generation_info = parsed_response if parsed_response.get("stop") is True else None
    return ChatGenerationChunk(
        message=AIMessageChunk(content=parsed_response.get("content", "")),
        generation_info=generation_info,
    )


class ChatLlamaCppServer(ProgaiChatModel, _LlamaCppServerCommon):
    base_url: str | None = None
    """Base url the model is hosted under."""

    jinja2_prompt_template: str = (
        "{% for m in messages %}"
        "{% if m.type=='system' %}<|system|>"
        "{% elif m.type=='human' %}<|user|>"
        "{% else %}<|assistant|>{% endif %}"
        "\n{{ m.content }}</s> \n{% endfor %}"
        "<|assistant|>\n{{ start }}"
    )

    ai_message_start: str = ""
    """Force ai to start with specific string."""

    @root_validator(allow_reuse=True)
    def validate_endpoint(cls, values: Dict) -> Dict:
        """If no base_url is set explicitly, try to fallback to configuration file or environment variables."""
        values["base_url"] = values["base_url"] or get_endpoint("ZEPHYR7B")
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "zephyr-chat"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    def _prompt_from_template(self, messages: List[BaseMessage]) -> str:
        return (
            jinja2.Environment()
            .from_string(self.jinja2_prompt_template)
            .render(messages=messages, start=self.ai_message_start)
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._prompt_from_template(messages)
        final_chunk = super()._stream_with_aggregation(
            prompt, stop=stop, run_manager=run_manager, verbose=self.verbose, **kwargs
        )
        chat_generation = ChatGeneration(
            message=AIMessage(content=final_chunk.text),
            generation_info=final_chunk.generation_info,
        )
        return ChatResult(generations=[chat_generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prompt = self._prompt_from_template(messages)
        for stream_resp in self._create_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_chat_generation_chunk(stream_resp)
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        prompt = self._prompt_from_template(messages)
        async for stream_resp in self._acreate_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_chat_generation_chunk(stream_resp)
                yield chunk
                if run_manager:
                    await run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
