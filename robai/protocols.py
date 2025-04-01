from typing import List, Optional, Any
from fastapi import WebSocket
from robai.schemas import (
    NewMessageEvent,
    MessageChunkEvent,
    MessageCompleteEvent,
    RobotStatusUpdateEvent,
    ErrorEvent,
    RobaiEvent,
    ChatMessage,
)
import asyncio
from rich.console import Console
import shutil
import json
from abc import ABC
from loguru import logger
import time
from rich.markdown import Markdown
from rich.syntax import Syntax
import re
from rich.panel import Panel


def format_markdown(text: str) -> None:
    """Format and print markdown text to the console."""
    md = Markdown(text)
    return str(md)


def format_json(obj: Any) -> str:
    """Format JSON data with syntax highlighting."""
    json_str = json.dumps(obj, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai")
    return str(syntax)


# MESSAGE HANDLER
class MessageHandler(ABC):
    """Base class for handling I/O operations."""

    def __init__(self, robot_name: str = None):
        """Initialize the message handler."""
        self._robot_name = robot_name
        self._current_robot = None
        self.history: List[ChatMessage] = []

    @property
    def robot_name(self) -> str:
        """Dynamically get the robot name from the current context."""
        # If a robot is currently using this handler, use its name
        if self._current_robot:
            return self._current_robot.__class__.__name__
        # Otherwise fall back to the default name
        return self._robot_name or "AI"

    @robot_name.setter
    def robot_name(self, value: str) -> None:
        """Set the default robot name."""
        self._robot_name = value

    def set_current_robot(self, robot) -> None:
        """Set the current robot using this handler."""
        self._current_robot = robot

    def clear_current_robot(self) -> None:
        """Clear the current robot reference."""
        self._current_robot = None

    # Core I/O - must implement
    async def get_input(self) -> ChatMessage:
        """Get input from the source"""
        raise NotImplementedError

    async def send_new_message(self, name: str, message_id: str) -> None:
        """Signal start of new message from this robot"""
        raise NotImplementedError

    async def send_chunk(self, content: str, message_id: str) -> None:
        """Send a chunk of message content (for streaming)"""
        raise NotImplementedError

    async def send_message_complete(self, message_id: str) -> None:
        """Signal that the message is complete"""
        raise NotImplementedError

    async def send_error(self, error: str) -> None:
        """Handle any errors during message processing"""
        raise NotImplementedError

    async def update_status(self, message: str, is_complete: bool = False) -> None:
        """Update status in the appropriate medium"""
        raise NotImplementedError

    # Default in-memory history implementation
    async def save_to_history(self, message: ChatMessage) -> None:
        """Default implementation stores in memory"""
        self.history.append(message)

    async def get_history(self, limit: int = 10) -> List[ChatMessage]:
        """Default implementation returns from memory"""
        return self.history[-limit:]

    async def clear_history(self) -> None:
        """Default implementation clears memory"""
        self.history.clear()

    async def wait_for_input_ready(self) -> None:
        """Default implementation does nothing"""
        pass

    async def simulate_message(
        self, message: str, chunk_size: int = 3, delay: float = 0.05
    ) -> None:
        """Simulate a realistic message stream with chunking and delays.

        Args:
            message: The full message to simulate
            chunk_size: How many words to send per chunk
            delay: Delay between chunks in seconds
        """
        await self.send_new_message()

        # Split into words and chunk them
        words = message.split()
        chunks = [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

        # Send each chunk with a delay
        for i, chunk in enumerate(chunks):
            # Add space if not the first chunk
            if i > 0:
                chunk = " " + chunk
            await self.send_chunk(chunk)
            await asyncio.sleep(delay)  # Simulate typing/thinking delay

        await self.send_message_complete()


# Concrete implementations
class ConsoleMessageHandler(MessageHandler):
    """Console implementation with default in-memory history and proper message streaming"""

    def __init__(self):
        super().__init__()
        self.console = Console()
        self._last_status = None
        self._last_status_time = 0
        self._status_update_threshold = 1.0
        self._current_message = ""
        self._is_streaming = False
        self.width = shutil.get_terminal_size().columns - 2

    async def send_new_message(self, name: str, message_id: str) -> None:
        """Start a new message."""
        self._is_streaming = True
        self._current_message = ""
        print(f"\n🤖 {name}: {message_id}", end=" ", flush=True)

    async def send_chunk(self, chunk: str, message_id) -> None:
        """Use simple print for streaming chunks"""
        try:
            self._current_message += chunk
            print(chunk, end="", flush=True)
        except Exception as e:
            logger.error(f"Error in send_chunk: {e}")
            print(chunk, end="", flush=True)

    async def send_message_complete(self, message_id: str) -> None:
        """Complete the message with a single newline"""
        self._is_streaming = False
        print()  # Simple newline

    async def wait_for_input(self) -> ChatMessage:
        """Get user input and echo it consistently"""
        text = await self.get_input()
        # Use Rich for the user message formatting
        self.console.print(f"\n👤 User: {text}", style="bright_white")
        return ChatMessage(role="user", content=text)

    async def get_input(self, prompt: str = None) -> str:
        """Get input from the user with proper spacing"""
        if prompt:
            return input(f"\n❓ {prompt}: ")
        # Remove the explicit User: prefix since we'll echo it after
        return input("\n> ")

    async def update_status(self, status: str) -> None:
        """Update status with better spacing and context"""
        cleaned_status = re.sub(r"\[/?[a-zA-Z]+\]", "", status)
        cleaned_status = cleaned_status.replace("[", "\\[").replace("]", "\\]")

        current_time = time.time()
        if (
            cleaned_status != self._last_status
            and current_time - self._last_status_time > self._status_update_threshold
        ):
            # Format different types of status messages
            if "Calling" in cleaned_status:
                if hasattr(self, "_last_function_args") and self._last_function_args:
                    args_str = ", ".join(
                        f"{k}={v}" for k, v in self._last_function_args.items()
                    )
                    status_msg = f"📝 Calling function: {cleaned_status.split('Calling')[1].strip()}({args_str})"
                else:
                    status_msg = f"📝 Calling function: {cleaned_status.split('Calling')[1].strip()}"
            elif "Function returned" in cleaned_status:
                status_msg = f"🔄 {cleaned_status}"
            else:
                status_msg = f"📝 {cleaned_status}"

            self.console.print(f"\n{status_msg}", style="bold yellow")
            self._last_status = cleaned_status
            self._last_status_time = current_time

    async def handle_function_call(self, function_name: str, arguments: dict) -> None:
        """Store function arguments for status updates"""
        self._last_function_args = arguments
        # Format and display the function call
        args_str = ", ".join(f"{k}={v}" for k, v in arguments.items())
        self.console.print(
            f"\n📝 Calling {function_name}({args_str})", style="bold yellow"
        )

    async def handle_function_result(self, function_name: str, result: Any) -> None:
        """Enhanced function result handling with status updates"""
        # Show completion status
        await self.update_status(f"✅ Completed {function_name}")

        # Then show the result panel
        self.console.print(
            Panel.fit(
                f"[bold green]✅ {function_name}[/]\n{format_json(result)}",
                title="Function Result",
                border_style="green",
                padding=(0, 2),
            )
        )


class WebSocketMessageHandler(MessageHandler):
    """Websocket implementation.
    Note: This is a basic implementation - for database history,
    extend this class in your application layer. Example:

    ```python
    # In your application code (e.g., app/ai/handlers.py):
    class DatabaseWebSocketHandler(WebSocketHandler):
        async def save_to_history(self, message: ChatMessage) -> None:
            async with your_db_session() as db:
                # Save to your database
                conversation = await self._get_conversation(db)
                message_record = YourMessageModel(
                    role=message.role,
                    content=message.content,
                    conversation_id=conversation.id
                )
                db.add(message_record)
                await db.commit()

        async def get_history(self, limit: int = 10) -> List[ChatMessage]:
            async with your_db_session() as db:
                # Fetch from your database
                messages = await db.execute(
                    select(YourMessageModel)
                    .where(YourMessageModel.conversation_id == self.conversation_id)
                    .order_by(YourMessageModel.timestamp.desc())
                    .limit(limit)
                )
                return [
                    ChatMessage(role=msg.role, content=msg.content)
                    for msg in reversed(messages.scalars().all())
                ]
    ```
    """

    def __init__(self, websocket: WebSocket, robot_name: str):
        super().__init__(robot_name=robot_name)
        self.websocket = websocket
        self.event_lock = asyncio.Lock()

    async def wait_for_input(self) -> ChatMessage:
        """Wait for input from websocket, handling events"""
        while True:
            data = await self.websocket.receive_text()
            json_data = json.loads(data)

            if "role" in json_data and "content" in json_data:
                msg = ChatMessage(role=json_data["role"], content=json_data["content"])
                return msg
            elif "event" in json_data:
                # Handle any pre-input events (like setup)
                await self._handle_event(RobaiEvent(**json_data))

    async def _handle_event(self, event: RobaiEvent) -> None:
        """Handle various frontend events"""
        # Override this in subclasses to handle specific events
        pass

    async def update_status(self, message: str, is_complete: bool = False) -> None:
        """Send status update to frontend"""
        async with self.event_lock:
            await self.websocket.send_text(
                RobotStatusUpdateEvent(
                    data={"status": message, "is_complete": is_complete}
                ).model_dump_json()
            )

    async def send_new_message(self, name: str, message_id: str) -> None:
        """Signal start of new message to frontend using this robot's name"""
        async with self.event_lock:
            await self.websocket.send_text(
                NewMessageEvent(
                    data=NewMessageEvent.Data(
                        name=self.robot_name, message_id=message_id
                    )
                ).model_dump_json()
            )

    async def send_chunk(self, content: str, message_id: str) -> None:
        """Stream message chunk to frontend"""
        async with self.event_lock:
            await self.websocket.send_text(
                MessageChunkEvent(
                    data=MessageChunkEvent.Data(
                        content=content, message_id=message_id
                    ).model_dump_json()
                )
            )

    async def send_message_complete(self, message_id: str) -> None:
        """Signal message completion to frontend"""
        async with self.event_lock:
            await self.websocket.send_text(
                MessageCompleteEvent(
                    data=MessageCompleteEvent.Data(message_id=message_id)
                ).model_dump_json()
            )

    async def send_error(self, error: str) -> None:
        """Send error event to frontend"""
        async with self.event_lock:
            await self.websocket.send_text(ErrorEvent(data=error).model_dump_json())


class RobotHandler(MessageHandler):
    """Robot-to-robot communication with default in-memory history"""

    def __init__(self, target_robot, robot_name: str = None):
        super().__init__(robot_name=robot_name)
        self.target_robot = target_robot

    async def get_input(self) -> ChatMessage:
        return self.target_robot.output_data

    async def update_status(self, message: str, is_complete: bool = False) -> None:
        logger.info(f"Robot status: {message}")
