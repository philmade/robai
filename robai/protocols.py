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


class WebSocketMessageHandler(MessageHandler):
    """Websocket implementation using standardized payloads.
    Assumes the connected WebSocket expects events matching app.schemas.ws_events.
    """

    try:
        from app.schemas.ws_events import (
            WSEvent as AppWSEvent,
            EventType as AppEventType,
            PayloadUnion as AppPayloadUnion,
        )

        _ws_event_imports_ok = True
    except ImportError:
        logger.warning(
            "Could not import app.schemas.ws_events. WebSocketMessageHandler may not function correctly."
        )
        AppWSEvent = None
        AppEventType = str
        AppPayloadUnion = Any
        _ws_event_imports_ok = False

    def __init__(self, websocket: WebSocket, robot_name: str = None):
        if not self._ws_event_imports_ok:
            raise ImportError(
                "Cannot initialize WebSocketMessageHandler: Failed to import required schemas from app.schemas.ws_events"
            )
        super().__init__(robot_name=robot_name)
        self.websocket = websocket
        self.event_lock = asyncio.Lock()
        self._current_robot_info = {"id": None, "name": None}

    def set_current_robot(self, robot) -> None:
        super().set_current_robot(robot)
        self._current_robot_info["id"] = getattr(
            robot, "robot_id", robot.__class__.__name__
        )
        self._current_robot_info["name"] = getattr(
            robot, "robot_name", robot.__class__.__name__
        )

    async def _send_event(self, event_type: AppEventType, payload: BaseModel):
        """Helper to send a structured WSEvent over the WebSocket."""
        async with self.event_lock:
            try:
                event = self.AppWSEvent(event_type=event_type, payload=payload)
                await self.websocket.send_json(event.model_dump(mode="json"))
            except Exception as e:
                logger.exception(
                    f"WebSocket send error: Failed to send event '{event_type}' - {e}"
                )

    async def stream_start(self, payload: StreamStartPayload) -> None:
        """Send stream_start event."""
        await self._send_event("stream_start", payload)

    async def stream_chunk(self, payload: StreamChunkPayload) -> None:
        """Send stream_chunk event."""
        await self._send_event("stream_chunk", payload)

    async def stream_end(self, payload: StreamEndPayload) -> None:
        """Send stream_end event."""
        await self._send_event("stream_end", payload)

    async def send_full_message(self, payload: RobaiChatMessagePayload) -> None:
        """Send chat_message event."""
        if payload.role == "assistant" and not payload.robot_id:
            payload.robot_id = self._current_robot_info.get("id") or self.robot_name
        if payload.role == "assistant" and not payload.robot_name:
            payload.robot_name = self._current_robot_info.get("name") or self.robot_name
        await self._send_event("chat_message", payload)

    async def update_status(self, payload: RobotStatusUpdatePayload) -> None:
        """Send robot_status_update event."""
        if not payload.robot_id:
            payload.robot_id = self._current_robot_info.get("id") or self.robot_name
        if not payload.robot_name:
            payload.robot_name = self._current_robot_info.get("name") or self.robot_name
        await self._send_event("robot_status_update", payload)

    async def send_error(self, payload: ErrorPayload) -> None:
        """Send error event."""
        await self._send_event("error", payload)

    async def wait_for_input(self) -> ChatMessage:
        from fastapi import WebSocketDisconnect

        while True:
            try:
                data = await self.websocket.receive_text()
                json_data = json.loads(data)

                if "event_type" in json_data and "payload" in json_data:
                    payload_data = json_data["payload"]
                    if (
                        json_data["event_type"] == "group_message"
                        and "content" in payload_data
                    ):
                        return ChatMessage(
                            role="user",
                            content=payload_data.get("content", ""),
                            name=payload_data.get("user_name"),
                            user_id=payload_data.get("user_id"),
                            chat_id=payload_data.get("chat_id"),
                            client_message_id=payload_data.get("client_message_id"),
                        )
                elif "role" in json_data and "content" in json_data:
                    return ChatMessage(**json_data)
                else:
                    logger.warning(
                        f"Received unexpected WebSocket message format: {json_data}"
                    )
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected while waiting for input.")
                raise
            except json.JSONDecodeError:
                logger.warning("Received invalid JSON over WebSocket.")
            except Exception as e:
                logger.exception(f"Error receiving WebSocket message: {e}")

    async def _handle_event(self, event: Any) -> None:
        """Placeholder for handling non-input events from client if needed."""
        logger.debug(f"WebSocketHandler received event: {event}")
        pass


class RobotHandler(MessageHandler):
    """Robot-to-robot communication. Adapts to new protocol.
    Primarily used for getting input FROM a target robot's output.
    Sending messages TO the target might need specific methods on the target.
    """

    def __init__(self, target_robot, robot_name: str = None):
        super().__init__(robot_name=robot_name)
        self.target_robot = target_robot

    async def get_input(self) -> ChatMessage:
        output_maybe_awaitable = self.target_robot.output()
        if asyncio.iscoroutine(output_maybe_awaitable):
            output = await output_maybe_awaitable
        else:
            output = output_maybe_awaitable

        if isinstance(output, (ChatMessage, AIMessage, SystemMessage)):
            return output
        elif isinstance(output, str):
            return AIMessage(
                role="assistant",
                content=output,
                name=self.target_robot.__class__.__name__,
            )
        elif isinstance(output, BaseModel):
            try:
                return AIMessage(**output.model_dump())
            except Exception:
                logger.warning(
                    f"Cannot convert target robot output ({type(output)}) to AIMessage."
                )
                return SystemMessage(
                    role="system",
                    content=f"Received complex output type: {type(output)}",
                )
        else:
            logger.warning(f"Unhandled target robot output type: {type(output)}")
            return SystemMessage(
                role="system", content=f"Received unhandled output type: {type(output)}"
            )

    async def stream_start(self, payload: StreamStartPayload) -> None:
        logger.debug(
            f"RobotHandler ({self.robot_name}): Ignoring stream_start for target {self.target_robot.__class__.__name__}"
        )
        pass

    async def stream_chunk(self, payload: StreamChunkPayload) -> None:
        logger.debug(
            f"RobotHandler ({self.robot_name}): Ignoring stream_chunk for target {self.target_robot.__class__.__name__}"
        )
        pass

    async def stream_end(self, payload: StreamEndPayload) -> None:
        logger.debug(
            f"RobotHandler ({self.robot_name}): Ignoring stream_end for target {self.target_robot.__class__.__name__}"
        )
        pass

    async def send_full_message(self, payload: RobaiChatMessagePayload) -> None:
        logger.debug(
            f"RobotHandler ({self.robot_name}): Ignoring send_full_message for target {self.target_robot.__class__.__name__}"
        )
        pass

    async def update_status(self, payload: RobotStatusUpdatePayload) -> None:
        logger.info(
            f"Robot Status ({payload.robot_name} - {payload.interaction_id[:6]}): {payload.status}"
        )

    async def send_error(self, payload: ErrorPayload) -> None:
        logger.error(
            f"Robot Error ({payload.interaction_id[:6]}): {payload.code} - {payload.message}"
        )
        if payload.context:
            logger.error(f"Error Context: {payload.context}")
