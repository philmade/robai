from pydantic import BaseModel
import tiktoken
from typing import List, Union, Dict, Optional

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


class TokenBaseModel(BaseModel):
    def token_count(self) -> int:
        return len(tokenizer.encode(self.model_dump_json()))


class ChatMessage(TokenBaseModel):
    role: str = "user"
    name: str = "case.bot"
    content: str = "Hello, I am a user"


class AIMessage(TokenBaseModel):
    role: str = "assistant"
    name: str = "case.bot"
    content: str = "Hello, I am an AI"


class SystemMessage(TokenBaseModel):
    role: str = "system"
    name: str = "case.bot"
    content: str = "Hello, I am a system message"


# Define base WebsocketEvent class
class RobaiEvent(BaseModel):
    event: str
    data: Union[str, list, BaseModel, dict, None] = None


# Define specific events with the nested data models
class NewMessageEvent(RobaiEvent):
    event: str = "newMessage"

    class Data(BaseModel):
        name: str  # The name of who is speaking (e.g. 'Assistant', 'User', 'Research Bot', etc)
        message_id: str

    data: Data


class MessageCompleteEvent(RobaiEvent):
    event: str = "messageComplete"

    class Data(BaseModel):
        message_id: str

    data: Data


class MessageChunkEvent(RobaiEvent):
    event: str = "newMessageChunk"

    class Data(BaseModel):
        message_id: str
        content: str

    data: str


class RobotStatusUpdateEvent(RobaiEvent):
    event: str = "robotStatusUpdate"

    class Data(BaseModel):
        status: str
        is_complete: bool

    data: Data


class ErrorEvent(RobaiEvent):
    event: str = "error"
    data: str


class MarkdownFunctionResults:
    def __init__(self):
        self.results = []
        self.errors = []

    def append(self, function_name: str, result: str):
        self.results.append(f"### {function_name}\n\n{result}")

    def add_error(self, error: str):
        self.errors.append(error)

    def __str__(self):
        return "\n\n".join(self.results)

    def is_empty(self) -> bool:
        return not self.results and not self.errors

    def has_results(self) -> bool:
        return bool(self.results or self.errors)

    def trim_oldest(self, max_results: int = 3) -> None:
        """Remove oldest results while keeping the most recent ones.
        Args:
            max_results: Maximum number of results to keep. Defaults to 3.
        """
        if len(self.results) > max_results:
            self.results = self.results[-max_results:]

    def clear(self) -> None:
        self.results = []
        self.errors = []


class FunctionCallFromAI(BaseModel):
    name: str
    arguments: Union[Dict, List, str, int]
