from robai.in_out import ChatMessage, AIMessage
from typing import List, Union, Any, Generator, Type
from pydantic import BaseModel


# Define the BaseMemory class
class BaseMemory(BaseModel):
    # The system prompt is the first message the robot recieves on initialisation, it's a ChatMessage
    # It is populated with the purpose string
    system_prompt: ChatMessage = ChatMessage(role="system", content="purpose")
    # The input model is what goes into your robot
    # input_model: Type[Any] = ChatMessage
    # The instructions for the AI is what will be sent to the AI. Build this in pre-call chain.
    instructions_for_ai: List[ChatMessage] = []
    # List of messages that have been sent / received by the AI
    message_history: List[ChatMessage] = []
    # When the memory.complete is true when the robot has finished its job
    complete: bool = False

    # The AI response will always be available as a ChatMessage object after the AI has been called
    ai_response: ChatMessage = None
    ai_raw_response: Any = None
    ai_string_response: str = None
    ai_response_generator: Generator = None

    # ROBOCALL
    # When two robots are talking to each other, the robo_call_complete is set to True
    # when the conversation is complete.
    in_robo_call: bool = False
    # When robots call each other directly, they store the response in robot_response
    robot_response: ChatMessage = None

    def add_message_to_history(self, message: Union[ChatMessage, AIMessage]):
        """
        Add a new message to the message_history and truncate based on tokens if necessary.
        """
        self.message_history.append(message)

    def set_complete(self, persist: bool = False):
        """Set the memory to complete."""
        self.complete = True
        if not persist:
            self.reset_memory()

    def forget_everything(self):
        """Reset attributes of the BaseMemory instance."""
        for field_name, field in self.__fields__.items():
            setattr(self, field_name, field.default)

    # PRE-CALL CHAIN 1
    def reset_memory(self):
        input_model = self.input_model
        self.forget_everything()
        self.input_model = input_model

    class Config:
        arbitrary_types_allowed = True
