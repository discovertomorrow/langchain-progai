# LangChain-ProgAI

![langchain_progai](.img/langchain_progai.png)

Explorative python package of [prognostica](https://www.prognostica.de/de/) providing [langchain](https://github.com/langchain-ai/langchain/tree/master)-conform classes to utilize open source LLMs on local infrastructure.


## Configuration

LangChain-ProgAI requires the configuration of model endpoints for interaction. This can be done by defining a yaml configuration file, such as the provided [default configuration](langchain_progai/config/default_config.yaml). To switch to a custom configuration file, as required when running langchain_progai outside prognosticas AI infrastructure, set an environment variable `LANGCHAIN_PROGAI_CONFIG` with the path pointing to your file.

As an alternative (or addition), it is possible to set individual endpoints via environment variables, overwriting possible configurations in a configuration file. By convention, for the `ZEPHYR7B` endpoint an corresponding environmentvariable would be defined as `ENDPOINT_ZEPHYR7B`.

Besides endpoint configuration, LangChain_ProgAI requires an environment variable `PROGAI_TOKEN` with a user or application specific token. The particular use of this token, e.g. for authentication or determination of model slots, depends on the LLM runtime in the backend (for details check out prognosticas [ProgAI middleware](https://github.com/discovertomorrow/progai-middleware/pkgs/container/progai-middleware)).

## Disclaimer

While developed by prognosticians, langchain-progai is not an official prognostica product.