from langchain_progai.chat.llamacpp_chat import ChatLlamaCppServer
from langchain_progai.config import get_endpoint


class ZephyrChat(ChatLlamaCppServer):
    base_url: str = get_endpoint("ZEPHYR7B")
    stop: list[str] | None = ["</s>", "<|user|>", "<|assistant|>", "<|system|>"]
    jinja2_prompt_template: str = (
        "{% for m in messages %}"
        "{% if m.type=='system' %}<|system|>"
        "{% elif m.type=='human' %}<|user|>"
        "{% else %}<|assistant|>{% endif %}"
        "\n{{ m.content }}</s>\n{% endfor %}"
        "<|assistant|>\n{{ start }}"
    )


class MixtralInstructChat(ChatLlamaCppServer):
    base_url: str = get_endpoint("MIXTRAL")
    stop: list[str] | None = ["</s>", "[INST]", "[/INST]"]
    jinja2_prompt_template: str = (
        "{% for m in messages %}"
        "{% if m.type=='system' %} [INST] {{ m.content }} [/INST]"
        "{% elif m.type=='human' %} [INST] {{ m.content }} [/INST]"
        "{% else %} {{ m.content }}</s>{% endif %}{% endfor %}"
        "{% if start %} {{ start }}{% endif %}"
    )


class Llama3Chat(ChatLlamaCppServer):
    base_url: str = get_endpoint("LLAMA3")
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
