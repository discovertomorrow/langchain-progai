import pytest
import os
import shutil
import logging
import langchain_progai.config as config
from importlib import resources


logger = logging.getLogger(__name__)


@pytest.fixture
def setup_test_config():
    """Fixture to make sure that at least a template config.yaml is available during test execution."""
    config_path = str(resources.files("langchain_progai") / "config/config.yaml")
    template_path = str(resources.files("langchain_progai") / "config/config.yaml.template")
    file_was_copied = False

    if not os.path.exists(config_path):
        shutil.copy(template_path, config_path)
        file_was_copied = True

    yield ...

    # Clean up: delete config.yaml if it was copied
    if file_was_copied and os.path.exists(config_path):
        os.remove(config_path)


def _assert_valid_config(conf: dict):
    assert isinstance(conf, dict)
    assert "endpoints" in conf
    assert len(conf["endpoints"]) > 0


def test_load_config___given_no_input___loads_default_config(setup_test_config):
    # Arrange & Act
    conf = config.load_config(path=None)

    # Assert
    _assert_valid_config(conf)


def test_load_config___given_wrong_path___returns_fallback(caplog, setup_test_config):
    # Arrange & Act
    conf = config.load_config(path="NonExistingPath.yaml")

    # Assert
    assert len(conf["endpoints"]) > 0
    assert "No configuration file found" in caplog.text


def test_load_config___given_path_in_environment_variable___works_as_expected(caplog, setup_test_config):
    # Arrange
    os.environ['LANGCHAIN_PROGAI_CONFIG'] = str(resources.files("langchain_progai") / "config/config.yaml")

    # Act
    conf = config.load_config(path=None)

    # Assert
    assert len(conf["endpoints"]) > 0
    assert "No configuration file found" not in caplog.text


def test_get_endpoint___given_endpoint_name___returns_endpoint_string(setup_test_config):
    # Arrange & Act
    endpoint = config.get_endpoint("DOLPHIN")

    # Assert
    assert isinstance(endpoint, str)


def test_get_endpoint___endpoint_in_env___overwrites_default_configuration(setup_test_config):
    # Arrange
    name = "DOLPHIN"
    test_endpoint = "http://test"
    os.environ[f"ENDPOINT_{name}"] = test_endpoint

    # Act
    endpoint = config.get_endpoint(name)

    # Assert
    assert endpoint == test_endpoint
