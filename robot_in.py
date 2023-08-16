from pydantic import BaseModel
from typing import Optional, List


class BaseRobotInput(BaseModel):
    def prepare_for_ai(self) -> str:
        """
        Implement a method to turn your models attributes into instructions for the AI model
        """
        raise NotImplementedError

    class Config:
        arbitrary_types_allowed = True


class BaseRobotOutput(BaseModel):
    pass


class SummaryInputModel(BaseRobotInput):
    summary_instructions: Optional[str]
    text: str

    def prepare_for_ai(self):
        return self.json()


class AnswerOut(BaseModel):
    answer: str = None


class ChatMessage(BaseModel):
    role: Optional[str] = "user"
    content: str = "Hello, I am a user"


class AIMessage(BaseModel):
    role: Optional[str] = "assistant"
    content: str = "Hello, I am a helpful AI"


class Conversation(BaseModel):
    history: List[ChatMessage] = []
