import os
import yaml
from typing import Callable
from importlib import resources
import logging


logger = logging.getLogger(__name__)


def load_config(path: str | None = None) -> dict | None:
    """Utility function to load a langchain_progai configuration with model endpoints.

    The function takes an hierarchical approach. Without specification, the projects default_config.yaml is loaded.
    If an environment variable LANGCHAIN_PROGAI_CONFIG exists, this is preferred, but overwritten from an explicit
    specification of path.

    Parameters
    ----------
    path
        Path of config.yaml to load.

    Returns
    -------
    Dictonary of loaded yaml.
    """
    def _existing(path):
        if path:
            if os.path.isfile(path):
                return path
            else:
                logger.warning(f"No configuration file found at specified location {path}")
                return None
        else:
            return None

    path = (
        _existing(path) or
        _existing(os.getenv("LANGCHAIN_PROGAI_CONFIG")) or
        _existing(resources.files("langchain_progai") / "config/config.yaml")
    )

    if not path:
        raise RuntimeError(
            "Found no valid endpoint configuration. Provide path to configuration file as input parameter or "
            "environment variable LANGCHAIN_PROGAI_CONFIG, "
            "or create default config at langchain_progai/config/config.yaml"
        )

    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    return config


def get_endpoint(
        name: str,
        name_to_env_pattern: Callable | None = lambda x: f"ENDPOINT_{x.upper()}",
    ) -> str | None:
    """Get endpoint from environment.

    Parameters
    ----------
    name
        Name of endpoint to retrieve.
    name_to_env_pattern
        Callable to transform endpoint name to the corresponding environment variable name.

    Returns
    -------
    Endpoint url.
    """

    return os.getenv(name_to_env_pattern(name) if name_to_env_pattern else name) or load_config()["endpoints"][name]
