from robai.base import BaseRobot
from robai.schemas import ChatMessage, SystemMessage
from robai.protocols import MessageHandler
from typing import Optional, TypeVar

OutputType = TypeVar("OutputType")


class ChatRobot(BaseRobot[ChatMessage, OutputType]):
    """
    ChatRobot is a base class for robots that communicate via ChatMessage objects.
    All chat-based robots should inherit from this class.
    """

    def __init__(
        self,
        message_handler: MessageHandler,
        ai_model: str = "gpt-4o-mini",
        stream: bool = True,
        max_tokens: Optional[int] = 4096,
        **kwargs,
    ):
        super().__init__(
            message_handler=message_handler,
            ai_model=ai_model,
            stream=stream,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Set a default system prompt
        self.system_prompt = SystemMessage(
            content="You are a helpful chat robot. Someone is now talking to you."
        )

    async def prepare(self) -> None:
        """Save message to history and update prompt"""
        await self.message_handler.save_to_history(self.input_data)
        history = await self.message_handler.get_history(10)
        self.prompt = [self.system_prompt] + history + [self.input_data]

    async def process(self) -> None:
        """Process the AI response with streaming"""
        await self._handle_streaming_response()

    async def finalize(self) -> None:
        """No default finalization needed"""
        pass

    async def stop_condition(self) -> bool:
        """Stop when finished flag is set"""
        return self.finished
