from .robot_in import ChatMessage, AIMessage
from typing import List, Union, Any, Type, TypeVar
from pydantic import BaseModel

TypeInputModel = TypeVar("TypeInputModel")
TypeOutputModel = TypeVar("TypeOutputModel")


# Define the BaseMemory class
class BaseMemory(BaseModel):
    purpose: str
    input_model: Type[Any] = None
    output_model: Type[Any] = None
    instructions_for_ai: Union[
        str, List[str], List[dict], List[bool], List[int], List[float], List[list]
    ] = None
    message_history: List[Union[ChatMessage, AIMessage]] = []
    max_message_history_length: int = 1000
    complete: bool = False
    prompt_as_string: str = None
    prompt_as_message: ChatMessage = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the message_history with the purpose message
        self.prompt_as_message = ChatMessage(role="system", content=self.purpose)
        self.add_message(self.prompt_as_message)
        self.prompt_as_string = self.purpose

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a string.
        """
        token_length = text.split().__len__() / 1.5
        return token_length

    def total_tokens_in_history(self) -> int:
        """Count the total number of tokens in the message_history."""
        return sum(
            self.count_tokens(message.content) for message in self.message_history
        )

    def add_message(self, message: Union[ChatMessage, AIMessage]):
        """Add a new message to the message_history and truncate based on tokens if necessary."""
        # while (
        #     self.total_tokens_in_history() + self.count_tokens(message.content)
        #     > self.max_message_history_length
        # ):
        #     self.message_history.pop(0)  # Remove the oldest message
        self.message_history.append(message)

    def set_complete(self):
        """Set the memory to complete."""
        self.complete = True

    class Config:
        arbitrary_types_allowed = True


# MEMORY MODULE
