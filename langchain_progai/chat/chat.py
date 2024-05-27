from langchain_progai.chat.llamacpp_chat import ChatLlamaCppServer
from langchain_progai.config import get_endpoint
from langchain_core.pydantic_v1 import root_validator

from typing import Dict


class ZephyrChat(ChatLlamaCppServer):
    base_url: str | None = None
    stop: list[str] | None = ["</s>", "<|user|>", "<|assistant|>", "<|system|>"]
    jinja2_prompt_template: str = (
        "{% for m in messages %}"
        "{% if m.type=='system' %}<|system|>"
        "{% elif m.type=='human' %}<|user|>"
        "{% else %}<|assistant|>{% endif %}"
        "\n{{ m.content }}</s>\n{% endfor %}"
        "<|assistant|>\n{{ start }}"
    )

    @root_validator(allow_reuse=True)
    def validate_endpoint(cls, values: Dict) -> Dict:
        """If no base_url is set explicitly, try to fallback to configuration file or environment variables."""
        values["base_url"] = values["base_url"] or get_endpoint("ZEPHYR7B")
        return values


class MixtralInstructChat(ChatLlamaCppServer):
    base_url: str | None = None
    stop: list[str] | None = ["</s>", "[INST]", "[/INST]"]
    jinja2_prompt_template: str = (
        "{% for m in messages %}"
        "{% if m.type=='system' %} [INST] {{ m.content }} [/INST]"
        "{% elif m.type=='human' %} [INST] {{ m.content }} [/INST]"
        "{% else %} {{ m.content }}</s>{% endif %}{% endfor %}"
        "{% if start %} {{ start }}{% endif %}"
    )

    @root_validator(allow_reuse=True)
    def validate_endpoint(cls, values: Dict) -> Dict:
        """If no base_url is set explicitly, try to fallback to configuration file or environment variables."""
        values["base_url"] = values["base_url"] or get_endpoint("MIXTRAL")
        return values


class Llama3Chat(ChatLlamaCppServer):
    base_url: str | None = None
    stop: list[str] | None = [
        "</s>",
        "<s>",
        "<|start_header_id|>",
        "<|begin_of_text|>",
        "<|eot_id|>",
        "<|end_header_id|>",
    ]
    jinja2_prompt_template: str = (
        "<|begin_of_text|>{% for m in messages %}"
        "<|start_header_id|>{% if m.type=='human' %}user{% else %}{{ m.type }}{% endif %}<|end_header_id|>\n\n"
        "{{ m.content }}<|eot_id|>{% endfor %}"
        "<|start_header_id|>assistant<|end_header_id|>\n\n{{ start }}"
    )

    @root_validator(allow_reuse=True)
    def validate_endpoint(cls, values: Dict) -> Dict:
        """If no base_url is set explicitly, try to fallback to configuration file or environment variables."""
        values["base_url"] = values["base_url"] or get_endpoint("LLAMA3")
        return values
