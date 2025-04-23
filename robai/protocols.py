from typing import List, Optional, Any, Coroutine, Union, Dict
from fastapi import WebSocket
from robai.schemas import (
    StreamStartPayload,
    StreamChunkPayload,
    StreamEndPayload,
    RobotStatusUpdatePayload,
    ErrorPayload,
    RobaiChatMessagePayload,
    ChatMessage,
    AIMessage,
    SystemMessage,
)
import asyncio
from rich.console import Console
import shutil
import json
from abc import ABC, abstractmethod
from loguru import logger
import time
from rich.markdown import Markdown
from rich.syntax import Syntax
import re
from rich.panel import Panel
from uuid import UUID
import uuid as uuid_pkg # Alias for generating new UUIDs if needed
from pydantic import BaseModel


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
    """Base class for handling I/O operations using standardized payloads."""

    def __init__(self, robot_name: str = None):
        """Initialize the message handler."""
        self._robot_name = robot_name
        self._current_robot = None
        self.history: List[Union[ChatMessage, AIMessage, SystemMessage]] = []

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
    @abstractmethod
    async def get_input(self) -> ChatMessage:
        """Get input from the source. Returns internal ChatMessage type."""
        raise NotImplementedError

    @abstractmethod
    async def stream_start(self, payload: StreamStartPayload) -> None:
        """Signal the start of a new message stream."""
        raise NotImplementedError

    @abstractmethod
    async def stream_chunk(self, payload: StreamChunkPayload) -> None:
        """Send a chunk of message content during a stream."""
        raise NotImplementedError

    @abstractmethod
    async def stream_end(self, payload: StreamEndPayload) -> None:
        """Signal the end of a message stream."""
        raise NotImplementedError

    @abstractmethod
    async def send_full_message(self, payload: RobaiChatMessagePayload) -> None:
        """Send a complete, non-streamed message."""
        raise NotImplementedError

    @abstractmethod
    async def update_status(self, payload: RobotStatusUpdatePayload) -> None:
        """Update status in the appropriate medium."""
        raise NotImplementedError

    @abstractmethod
    async def send_error(self, payload: ErrorPayload) -> None:
        """Handle and send structured errors."""
        raise NotImplementedError

    # Default in-memory history implementation
    async def save_to_history(
        self, message: Union[ChatMessage, AIMessage, SystemMessage]
    ) -> None:
        """Default implementation stores internal message types in memory"""
        self.history.append(message)

    async def get_history(
        self, limit: int = 10
    ) -> List[Union[ChatMessage, AIMessage, SystemMessage]]:
        """Default implementation returns internal message types from memory"""
        return self.history[-limit:]

    async def clear_history(self) -> None:
        """Default implementation clears memory"""
        self.history.clear()

    async def wait_for_input_ready(self) -> None:
        """Default implementation does nothing"""
        pass


# Concrete implementations
class ConsoleMessageHandler(MessageHandler):
    """Console implementation using standardized payloads and Rich formatting."""

    def __init__(self):
        super().__init__()
        self.console = Console()
        self._last_status = None
        self._last_status_time = 0
        self._status_update_threshold = 0.5
        self.width = shutil.get_terminal_size().columns - 2
        self._active_streams: Dict[str, str] = {}

    async def stream_start(self, payload: StreamStartPayload) -> None:
        """Start a new message stream in the console."""
        run_id = payload.interaction_id
        robot_name = payload.robot_name or payload.robot_id
        self.console.print(
            f"\n🤖 [bold cyan]{robot_name}[/] ({run_id[:6]}): ", end="", flush=True
        )
        self._active_streams[run_id] = ""

    async def stream_chunk(self, payload: StreamChunkPayload) -> None:
        """Print a chunk of message content for a specific stream."""
        run_id = payload.interaction_id
        if run_id in self._active_streams:
            self._active_streams[run_id] += payload.content
            print(payload.content, end="", flush=True)
        else:
            logger.warning(f"Received chunk for unknown stream: {run_id}")
            print(
                f"[Chunk for unknown {run_id[:6]}]: {payload.content}",
                end="",
                flush=True,
            )

    async def stream_end(self, payload: StreamEndPayload) -> None:
        """Complete the message stream in the console."""
        run_id = payload.interaction_id
        if run_id in self._active_streams:
            print()
            del self._active_streams[run_id]
        else:
            logger.warning(f"Received stream_end for unknown stream: {run_id}")
            print(f"\n[Stream End for unknown {run_id[:6]}]")

    async def send_full_message(self, payload: RobaiChatMessagePayload) -> None:
        """Send a complete, non-streamed message to the console."""
        role = payload.role
        name = (
            payload.robot_name
            or payload.user_name
            or payload.robot_id
            or payload.user_id
            or role
        )
        prefix = ""
        style = ""
        if role == "user":
            prefix = "👤"
            style = "bright_white"
        elif role == "assistant":
            prefix = (
                f"🤖 ({payload.interaction_id[:6]})" if payload.interaction_id else "🤖"
            )
            style = "bold cyan"
        elif role == "system":
            prefix = "⚙️"
            style = "dim"

        self.console.print(
            f"\n{prefix} [bold {style}]{name}[/]: {payload.content}", style=style
        )

    async def update_status(self, payload: RobotStatusUpdatePayload) -> None:
        """Update status with Rich formatting, including robot info."""
        status = payload.status
        robot_name = payload.robot_name or payload.robot_id
        run_id = payload.interaction_id

        current_time = time.time()
        if (
            status == self._last_status
            and current_time - self._last_status_time < self._status_update_threshold
        ):
            return

        status_prefix = "⏳"
        if "Calling" in status:
            status_prefix = "📞"
        elif "Error" in status or "Failed" in status:
            status_prefix = "❌"
        elif "Complete" in status or "Success" in status:
            status_prefix = "✅"

        status_msg = f"{status_prefix} [{robot_name} ({run_id[:6]})]: {status}"
        if payload.is_complete:
            status_msg += " (Complete)"

        print()
        self.console.print(f"{status_msg}", style="yellow")

        self._last_status = status
        self._last_status_time = current_time

    async def send_error(self, payload: ErrorPayload) -> None:
        """Display structured errors using Rich Panel."""
        title = f"❌ Error ({payload.code or 'GENERAL'})"
        if payload.interaction_id:
            title += f" Run: {payload.interaction_id[:6]}"

        error_content = f"[bold red]Message:[/bold red] {payload.message}"
        if payload.context:
            error_content += f"\n[bold]Context:[/bold]\n{format_json(payload.context)}"

        self.console.print(
            Panel(error_content, title=title, border_style="bold red", expand=False)
        )

    async def wait_for_input(self) -> ChatMessage:
        """Get user input and return internal ChatMessage type."""
        text = await self.get_input()
        user_payload = RobaiChatMessagePayload(
            chat_id=UUID("00000000-0000-0000-0000-000000000000"),
            role="user",
            user_name="User",
            content=text,
        )
        return ChatMessage(role="user", content=text, name="User")

    async def get_input(self, prompt: str = None) -> str:
        """Get input from the user."""
        prompt_text = f"\n❓ {prompt}: " if prompt else "\n> "
        return input(prompt_text)
