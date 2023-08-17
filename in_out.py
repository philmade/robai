from pydantic import BaseModel
from typing import Optional, List


class ChatMessage(BaseModel):
    role: Optional[str] = "user"
    content: str = "Hello, I am a user"


class AIMessage(BaseModel):
    role: Optional[str] = "assistant"
    content: str = "Hello, I am a helpful AI"


class Conversation(BaseModel):
    history: List[ChatMessage] = []
