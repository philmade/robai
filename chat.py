from robai.schemas import ChatMessage, AIMessage, SystemMessage
from typing import Union, List
from abc import ABC, abstractmethod
import uuid


class HistoryManager(ABC):
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.conversation_id = str(uuid.uuid4())
        self.conversation_history = []

    def create_message(self, role: str, content: str):
        if role == "user":
            return self.create_chat_message(content)
        elif role == "assistant":
            return self.create_ai_message(content)
        elif role == "system":
            return self.create_system_message(content)
        else:
            raise ValueError(f"Role {role} not recognized")

    def create_chat_message(self, content: str) -> ChatMessage:
        return ChatMessage(role="user", robot=self.robot_name, content=content)

    def create_ai_message(self, content: str) -> AIMessage:
        return AIMessage(role="assistant", robot=self.robot_name, content=content)

    def create_system_message(self, content: str) -> SystemMessage:
        return SystemMessage(role="system", robot=self.robot_name, content=content)

    @abstractmethod
    async def save_message_to_history(
        self, user_id: int, message: Union[ChatMessage, AIMessage, SystemMessage]
    ):
        pass

    @abstractmethod
    async def get_history(self, conversation_id: str, user_id: int, limit: int = 10):
        pass

    # @abstractmethod
    async def get_latest_conversation_id(self, user_id: int):
        pass

    # @abstractmethod
    async def get_latest_conversation(self, user_id: int, limit: int = 10):
        pass


class LocalHistoryManager(HistoryManager):
    def save_message_to_history(
        self, message: Union[ChatMessage, AIMessage, SystemMessage]
    ):
        self.conversation_history.append(message)

    def get_history(
        self, _slice: slice
    ) -> List[Union[ChatMessage, AIMessage, SystemMessage]]:
        history = self.conversation_history[_slice]
        return history

    def get_history_string(self, _slice: slice):
        history = self.get_history(_slice)
        return ["\n".join(message.model_dump_json()) for message in history]

    def get_history_count(self, _slice: slice):
        count = 0
        for message in self.get_history(_slice):
            count += message.token_count()
        return count

    def get_last_message(self):
        return self.conversation_history[-1]
