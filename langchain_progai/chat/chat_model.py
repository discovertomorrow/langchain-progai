from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from abc import abstractmethod
from typing import List


class ProgaiChatModel(BaseChatModel):
    @abstractmethod
    def _prompt_from_template(self, messages: List[BaseMessage]) -> str:
        "Prompt to string"

    def _prompt_debugger_lambda(self, input):
        print(self._prompt_from_template(input.messages))
        return input

    def get_prompt_debugger_runnable(self) -> RunnableLambda:
        return RunnableLambda(self._prompt_debugger_lambda)
