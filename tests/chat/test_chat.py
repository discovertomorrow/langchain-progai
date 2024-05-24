import pytest
from langchain_core.messages import AIMessage

import langchain_progai.chat as chat


@pytest.mark.integration
@pytest.mark.parametrize("cls", [
    chat.Llama3Chat,
    chat.MixtralInstructChat,
    chat.ZephyrChat
])
def test_invoke___given_input_string___returns_answers(cls):
    # Arrange
    llm = cls()

    # Act
    answer = llm.invoke("Who are you?")

    # Assert
    assert isinstance(answer, AIMessage)
    assert len(answer.content) > 0
