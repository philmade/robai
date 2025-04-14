from pydantic import BaseModel, Field
import tiktoken
from typing import List, Union, Dict, Optional, Any, Literal
import uuid
from uuid import UUID  # Import UUID directly for cleaner union syntax
from datetime import datetime, timezone

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


class TokenBaseModel(BaseModel):
    def token_count(self) -> int:
        # Ensure content exists and is a string before encoding
        content_str = ""
        if hasattr(self, "content") and isinstance(self.content, str):
            content_str = self.content
        elif isinstance(
            self, str
        ):  # Handle cases where the model itself might be string-like
            content_str = self
        # Consider other fields if needed, or serialize the whole model
        # For now, just focusing on content if available
        return len(tokenizer.encode(content_str))


class ChatMessage(TokenBaseModel):
    role: str = "user"
    name: Optional[str] = None  # Make optional as per app schema
    content: str = ""
    # Add fields from app schema for potential internal use/consistency
    user_id: Optional[Union[str, UUID]] = None
    chat_id: Optional[Union[str, UUID]] = None
    chat_title: Optional[str] = None
    client_message_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AIMessage(TokenBaseModel):
    role: str = "assistant"
    name: Optional[str] = None  # Make optional as per app schema
    content: str = ""
    # Add fields from app schema for potential internal use/consistency
    interaction_id: Optional[Union[str, UUID]] = None
    chat_id: Optional[Union[str, UUID]] = None
    robot_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SystemMessage(TokenBaseModel):
    role: str = "system"
    name: Optional[str] = None  # Make optional
    content: str = ""


# --- Standardized Payloads for MessageHandler Protocol ---
# Based on backend/app/schemas/ws_events.py


class RobaiChatMessagePayload(BaseModel):
    """Payload for sending a complete message (user, assistant, or system)."""

    message_id: Union[str, UUID] = Field(
        default_factory=lambda: str(uuid.uuid4())
    )  # Allow str/UUID
    interaction_id: Optional[Union[str, UUID]] = None
    chat_id: Optional[Union[str, UUID]] = None  # Context
    user_id: Optional[Union[str, UUID]] = None  # Sender's user ID (if user message)
    user_name: Optional[str] = None  # Sender's display name
    robot_id: Optional[str] = None  # Sender's robot ID (if robot message)
    robot_name: Optional[str] = None  # Sender's robot display name
    content: str
    role: Literal["user", "assistant", "system"]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    client_message_id: Optional[str] = None  # Original client ID if available
    metadata: Optional[Dict[str, Any]] = None  # Extra info


class RobotStatusUpdatePayload(BaseModel):
    """Payload for sending status updates about a robot's operation."""

    interaction_id: Union[str, UUID]
    robot_id: str
    robot_name: str
    chat_id: Union[str, UUID]  # Context for the update
    status: str  # e.g., "thinking", "processing_tool", "complete", "error"
    is_complete: bool = False


class ErrorPayload(BaseModel):
    """Payload for sending structured errors."""

    code: Optional[Union[str, int]] = None  # e.g., "ROBOT_ERROR", "INVALID_INPUT"
    message: str
    interaction_id: Optional[Union[str, UUID]] = None
    context: Optional[Dict[str, Any]] = None  # Optional context


class StreamStartPayload(BaseModel):
    """Payload signaling the start of a streamed response."""

    interaction_id: Union[str, UUID]
    robot_run_id: Union[str, UUID] = Field(  # Allow str/UUID
        alias="interaction_id"
    )  # Alias to interaction_id for backward compatibility
    robot_id: str  # ID of the robot sending the stream
    robot_name: str  # Display name of the robot
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    chat_id: Union[str, UUID]  # Context


class StreamChunkPayload(BaseModel):
    """Payload for a chunk of content within a stream."""

    interaction_id: Union[str, UUID]
    robot_run_id: Union[str, UUID] = Field(  # Allow str/UUID
        alias="interaction_id"
    )  # Alias to interaction_id for backward compatibility
    content: str  # The chunk of text


class StreamEndPayload(BaseModel):
    """Payload signaling the end of a streamed response."""

    interaction_id: Union[str, UUID]
    robot_run_id: Union[str, UUID] = Field(  # Allow str/UUID
        alias="interaction_id"
    )  # Alias to interaction_id for backward compatibility
    message_id: Optional[Union[str, UUID]] = None  # Allow str/UUID
    final_content: str  # The complete final message content


# --- Other Schemas (Keep if used internally) ---


class MarkdownFunctionResults:
    def __init__(self):
        self.results = []
        self.errors = []

    def append(self, function_name: str, result: str):
        self.results.append(f"### {function_name}\n\n{result}")

    def add_error(self, error: str):
        self.errors.append(f"ERROR: {error}")  # Prefix errors for clarity

    def __str__(self):
        # Combine results and errors into a single markdown string
        output = ""
        if self.results:
            output += "**Function Results:**\n" + "\n\n".join(self.results)
        if self.errors:
            if output:
                output += "\n\n"  # Add separator
            output += "**Function Errors:**\n" + "\n".join(
                f"- {e}" for e in self.errors
            )
        return output if output else "No function results or errors."

    def is_empty(self) -> bool:
        return not self.results and not self.errors

    def has_results(self) -> bool:
        return bool(self.results or self.errors)

    def trim_oldest(self, max_results: int = 3) -> None:
        if len(self.results) > max_results:
            self.results = self.results[-max_results:]
        # Decide if errors should also be trimmed or kept
        # if len(self.errors) > max_results:
        #     self.errors = self.errors[-max_results:]

    def clear(self) -> None:
        self.results = []
        self.errors = []


class FunctionCallFromAI(BaseModel):
    name: str
    arguments: Union[Dict, List, str, int, float, bool, None]  # Allow None argument
