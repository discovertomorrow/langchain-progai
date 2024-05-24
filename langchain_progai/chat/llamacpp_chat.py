import json
import jinja2
from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

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
    base_url: str = get_endpoint("ZEPHYR7B")
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
