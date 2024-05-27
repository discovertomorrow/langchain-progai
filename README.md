# LangChain-ProgAI

![langchain_progai](.img / langchain_progai.png)

Explorative python package of[prognostica](https: // www.prognostica.de / de/) providing[langchain](https: // github.com / langchain - ai / langchain / tree / master) - conform classes to utilize open source LLMs on local infrastructure.


# Configuration

LangChain - ProgAI requires the configuration of model endpoints for interaction. This can be done explicitly at the various classes requiring an endpoint(e.g. `langchain_progai.chat.ZephyrChat`) by setting a `base_url` or `endpoint` parameter at initialization.

Alternatively, this can be done by defining a yaml configuration file within the package at `langchain_progai/config/config.yaml`, or setting an environment variable `LANGCHAIN_PROGAI_CONFIG` with the path pointing to your file. [This template file](langchain_progai/config/config.yaml.template) contains a blueprint with possible endpoints to configure.

As a further alternative (or addition), it is possible to set individual endpoints via environment variables, overwriting possible configurations in a configuration file. By naming convention, e.g. an corresponding environment variable for the `ZEPHYR7B` endpoint would be defined as `ENDPOINT_ZEPHYR7B`.

Besides endpoint configuration, LangChain_ProgAI requires an environment variable `PROGAI_TOKEN` with a user or application specific token. The particular use of this token, e.g. for authentication or determination of model slots, depends on the LLM runtime in the backend (for details check out prognosticas [ProgAI middleware](https://github.com/discovertomorrow/progai-middleware/pkgs/container/progai-middleware)).

## Disclaimer

While developed by prognosticians, langchain-progai is not an official prognostica product.
