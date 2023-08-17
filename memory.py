from .in_out import ChatMessage, AIMessage
from typing import List, Union, Any, Type, TypeVar
from pydantic import BaseModel

TypeInputModel = TypeVar("TypeInputModel")
TypeOutputModel = TypeVar("TypeOutputModel")


# Define the BaseMemory class
class BaseMemory(BaseModel):
    # Purpose is a string that describes the purpose of the robot. It's a system prompt.
    purpose: str
    # The input model is what goes into your robot
    input_model: Type[Any] = ChatMessage
    # The instructions for the AI is what will be sent to the AI. Build this in pre-call chain.
    instructions_for_ai: Union[
        str, List[str], List[dict], List[bool], List[int], List[float], List[list]
    ] = None
    # List of messages that have been sent / received by the AI
    message_history: List[Union[ChatMessage, AIMessage]] = []
    # When the memory.complete is true when the robot has finished its job
    complete: bool = False

    # The AI response will always be available as a ChatMessage object after the AI has been called
    ai_response: ChatMessage = None

    # Other things that will be available:
    ai_raw_response: Any = None
    ai_string_response: str = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the message_history with the purpose message
        self.add_message(ChatMessage(role="system", content=f"{self.purpose}"))

    def add_message(self, message: Union[ChatMessage, AIMessage]):
        """
        Add a new message to the message_history and truncate based on tokens if necessary.
        """
        self.message_history.append(message)

    def set_complete(self):
        """Set the memory to complete."""
        self.complete = True

    class Config:
        arbitrary_types_allowed = True


# MEMORY MODULE
