import os
import logging
import langchain_progai.config as config
from importlib import resources


logger = logging.getLogger(__name__)


def _assert_valid_config(conf: dict):
    assert isinstance(conf, dict)
    assert "endpoints" in conf
    assert len(conf["endpoints"]) > 0


def test_load_config___given_no_input___loads_default_config():
    # Arrange & Act
    conf = config.load_config(path=None)

    # Assert
    _assert_valid_config(conf)


def test_load_config___given_wrong_path___returns_fallback(caplog):
    # Arrange & Act
    conf = config.load_config(path="NonExistingPath.yaml")

    # Assert
    assert len(conf["endpoints"]) > 0
    assert "No configuration file found" in caplog.text


def test_load_config___given_path_in_environment_variable___works_as_expected(caplog):
    # Arrange
    os.environ['LANGCHAIN_PROGAI_CONFIG'] = str(resources.files("langchain_progai") / "config/default_config.yaml")

    # Act
    conf = config.load_config(path=None)

    # Assert
    assert len(conf["endpoints"]) > 0
    assert "No configuration file found" not in caplog.text


def test_get_endpoint___given_endpoint_name___returns_endpoint_string():
    # Arrange & Act
    endpoint = config.get_endpoint("DOLPHIN")

    # Assert
    assert isinstance(endpoint, str)
    assert endpoint.startswith("http")


def test_get_endpoint___endpoint_in_env___overwrites_default_configuration():
    # Arrange
    name = "DOLPHIN"
    test_endpoint = "http://test"
    os.environ[f"ENDPOINT_{name}"] = test_endpoint

    # Act
    endpoint = config.get_endpoint(name)

    # Assert
    assert endpoint == test_endpoint
