from typing import (
    AsyncGenerator,
    List,
    Self,
    Any,
    TypeVar,
    Generic,
    Union,
    Callable,
    Iterable,
    Tuple,
    Dict,
    Optional,
    Literal,
)
from uuid import UUID # Ensure UUID is imported
import uuid as uuid_pkg # Alias for generating new UUIDs if needed
from openai import AsyncOpenAI
import openai
from openai.types.chat.chat_completion import ChatCompletion
import os
from abc import ABC, abstractmethod
from loguru import logger
from robai.schemas import (
    RobaiChatMessagePayload,
    SystemMessage,
    MarkdownFunctionResults,
    RobotStatusUpdatePayload,
    ErrorPayload,
    RobaiChatMessagePayload,
    StreamStartPayload,
    StreamEndPayload,
    StreamChunkPayload,
)
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from robai.utility import configure_logger, format_exc

from robai.protocols import (
    MessageHandler,
    ConsoleMessageHandler,
    ConsoleMessageHandler,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCallFunction,
    ChatCompletionChunk,
    ChoiceDeltaToolCall,
    ChoiceDelta,
    Choice,
)
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
import json
import traceback
from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from rich.panel import Panel
from rich.table import Table
import time
from dataclasses import dataclass
import tiktoken
import inspect
from datetime import datetime, timezone
import uuid as uuid_pkg

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


@dataclass
class PromptManager:
    """Manages a rolling window of chat messages with automatic trimming based on limits."""

    max_messages: int = 50  # Maximum number of messages to keep
    max_tokens: int = 4000  # Maximum total tokens allowed
    system_messages: List[SystemMessage] = None  # System messages are preserved
    messages: List[RobaiChatMessagePayload] = None  # Regular messages get trimmed
    tokenizer: Any = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def __post_init__(self):
        self.system_messages = self.system_messages or []
        self.messages = self.messages or []

    def set_system_prompt(self, message: SystemMessage) -> None:
        """Add a system message that will be preserved."""
        self.system_messages = [message]

    def add_message(self, message: RobaiChatMessagePayload) -> None:
        """Add a message to the window, automatically trimming if limits are exceeded."""
        self.messages.append(message)
        self._trim_if_needed()

    def _trim_if_needed(self) -> None:
        """Remove oldest messages until within limits."""
        # First check message count
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

        # Then check token count
        while self.total_tokens > self.max_tokens and len(self.messages) > 1:
            self.messages.pop(0)  # Remove oldest message

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens in all messages."""
        all_text = " ".join(msg.content for msg in self.system_messages + self.messages)
        return len(self.tokenizer.encode(all_text))

    @property
    def all_messages(self) -> List[Union[SystemMessage, RobaiChatMessagePayload]]:
        """Get all messages in order (system messages first)."""
        return self.system_messages + self.messages

    def clear_messages(self) -> None:
        """Clear all non-system messages."""
        self.messages = []

    def clear_all(self) -> None:
        """Clear all messages including system messages."""
        self.system_messages = []
        self.messages = []


class BaseAI:
    def __init__(
        self,
        message_handler: Optional[MessageHandler] = ConsoleMessageHandler(),
    ):
        API_KEY = os.getenv("OPENAI_API_KEY")
        if API_KEY is None:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.openai = openai
        self.openai.api_key = API_KEY
        self.aclient: AsyncOpenAI = AsyncOpenAI(api_key=API_KEY)
        self.models: List = self.aclient.models.list()
        self.model: str = "gpt-3.5-turbo"
        self.output_date: Union[str, AsyncGenerator[str, None]] = None
        self.stream: bool = False
        self.max_tokens: int = 100
        self.message_handler = message_handler
        self.testing = False

    async def generate_response(
        self,
        prompt_messages: List[RobaiChatMessagePayload],
        functions_for_ai: Dict[str, Callable] = None,
        force_function: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Tuple[Union[AsyncGenerator[str, None], str], Union[Iterable, None]]:
        messages = [message.model_dump() for message in prompt_messages]

        # Ensure only 'role' and 'content' keys are in the dictionaries
        messages = [
            {k: v for k, v in message.items() if k in ["role", "content"]}
            for message in messages
        ]
        if functions_for_ai:
            functions = [
                function.description() for function in functions_for_ai.values()
            ]
        else:
            functions = None

        # Set function_call parameter
        if force_function:
            tool_choice = {"function": {"name": force_function}, "type": "function"}
        elif functions:
            tool_choice = "auto"
        else:
            tool_choice = None

        # Build parameters dictionary
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "n": 1,
            "stop": None,
            "temperature": 0.7,
            "stream": self.stream,
        }

        # Only add tools-related parameters if functions exist
        if functions:
            params["tools"] = functions
            params["tool_choice"] = tool_choice

            params["parallel_tool_calls"] = False

        if self.stream:
            response_generator = await self.aclient.chat.completions.create(**params)
            return response_generator
        else:
            response: ChatCompletion = await self.aclient.chat.completions.create(
                **params
            )
            return (
                response.choices[0].message.content,
                response.choices[0].message.tool_calls,
            )


class BaseRobot(ABC, Generic[InputType, OutputType]):
    def __init__(
        self,
        message_handler: Optional[MessageHandler] = ConsoleMessageHandler(),
        ai_model: str = "gpt-4o-mini",
        stream: bool = True,
        max_tokens: int = 100,
        log_level: str = "WARNING",
        owns_message_handler: bool = True,
        interaction_id: Optional[str] = None,
        chat_id: Optional[UUID] = None,
        **kwargs,
    ) -> Self:
        """Initialize the robot.

        Args:
            message_handler: Handler for I/O operations
            ai_model: Model to use for AI operations
            stream: Whether to stream responses
            max_tokens: Maximum tokens for responses
            log_level: Logging level
            owns_message_handler: Whether this robot owns and should manage the message handler lifecycle
        """
        configure_logger(log_level)
        self.robot_id: str = self.__class__.__name__
        self.robot_name: str = self.__class__.__name__
        self.robot_id = kwargs.get("robot_id", self.robot_id)
        self.robot_name = kwargs.get("robot_name", self.robot_name)
        self.message_handler = message_handler
        self.message_handler.set_current_robot(self)
        self.owns_message_handler = owns_message_handler

        self.input_data: InputType = None
        self.output_data: OutputType = None
        self.stream = stream
        self.finished: bool = False

        # Chat and Interaction IDs
        # these may need to get set more granularly during a single call or interact() session
        self.interaction_id: str = interaction_id or str(uuid_pkg.uuid4())
        self.chat_id: Optional[UUID] = chat_id

        # Initialize AI
        self.ai_class = BaseAI(
            message_handler=message_handler,
        )
        self.ai_class.model = ai_model
        self.ai_class.stream = stream
        self.ai_class.max_tokens = max_tokens

        # Initialize state and prompt manager
        self.prompt_manager = PromptManager(
            max_messages=kwargs.get("max_messages", 50),
            max_tokens=kwargs.get("max_tokens", 4000),
        )
        self.system_prompt: Optional[SystemMessage] = None
        self._init_system_prompt()  # Initialize system prompt

        # Function management - collect all robot functions including inherited ones
        self.all_functions = {}
        for name, func in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(func, "is_robot_function"):
                self.all_functions[name] = func

        # Initialize available functions (only those marked as available)
        self.available_functions = {
            name: func
            for name, func in self.all_functions.items()
            if getattr(func, "available", True)
        }
        # Initialize restricted functions (empty by default)
        self.restricted_functions = {}

        self.force_function: Optional[str] = None
        self.function_results: MarkdownFunctionResults = MarkdownFunctionResults()
        self.skip_call: bool = False
        # State management
        self._accumulated_message = ""
        self._current_function = None
        self.pending_function_calls = []
        self.testing = False
        self.in_call = False
        self.calling_robot = None
        self.call_system_prompt = None

        # Robot call attributes
        self.connected_robot: Optional["BaseRobot"] = None
        self.is_call_initiator: bool = False

        # New state variables
        self.debug_context = kwargs.get("debug_context", True)
        self.max_tokens = kwargs.get("max_tokens", 5000)
        self.testing = kwargs.get("testing", False)
        self.skip_call = False
        self.finished: bool = False

        # Add action history tracking
        self.action_history: List[Tuple[datetime, str]] = []
        self.max_history_items = kwargs.get("max_history_items", 10)
        self.action_counts = {}  # Track frequency of actions
        self.error_count = 0  # Track recent errors

    # --- Payload Creation Helper Methods ---

    def _get_current_chat_id(self) -> UUID:
        """Returns the current chat_id or a fallback UUID."""
        return self.chat_id or uuid_pkg.UUID("00000000-0000-0000-0000-000000000000")

    def _create_status_payload(self, status: str, is_complete: bool = False) -> RobotStatusUpdatePayload:
        """Creates a RobotStatusUpdatePayload with current context."""
        return RobotStatusUpdatePayload(
            interaction_id=self.interaction_id,
            robot_id=self.robot_id,
            robot_name=self.robot_name,
            chat_id=self._get_current_chat_id(),
            status=status,
            is_complete=is_complete,
        )

    def _create_error_payload(self, message: str, code: Optional[str] = None, context: Optional[dict] = None) -> ErrorPayload:
        """Creates an ErrorPayload with current context."""
        return ErrorPayload(
            interaction_id=self.interaction_id,
            robot_id=self.robot_id, # Add robot_id for context
            chat_id=self._get_current_chat_id(), # Add chat_id for context
            message=message,
            code=code,
            context=context,
        )

    def _create_stream_start_payload(self) -> StreamStartPayload:
        """Creates a StreamStartPayload with current context."""
        return StreamStartPayload(
            interaction_id=self.interaction_id,
            robot_id=self.robot_id,
            robot_name=self.robot_name,
            chat_id=self._get_current_chat_id(),
        )

    def _create_stream_chunk_payload(self, content: str, message_id: Optional[str] = None) -> StreamChunkPayload:
        """Creates a StreamChunkPayload."""
        # message_id might be needed if consolidating chunks client-side based on a final message ID
        return StreamChunkPayload(
            interaction_id=self.interaction_id,
            content=content,
            message_id=message_id, # Pass along if provided
        )

    def _create_stream_end_payload(self, final_content: Optional[str] = None) -> StreamEndPayload:
        """Creates a StreamEndPayload."""
        return StreamEndPayload(
            interaction_id=self.interaction_id,
            final_content=final_content,
        )

    def _create_full_message_payload(self, content: str, role: str = "assistant", metadata: Optional[dict] = None) -> RobaiChatMessagePayload:
        """Creates a RobaiChatMessagePayload for sending a complete message."""
        return RobaiChatMessagePayload(
            message_id=str(uuid_pkg.uuid4()), # Generate new ID for the message
            interaction_id=self.interaction_id,
            chat_id=self._get_current_chat_id(),
            robot_id=self.robot_id if role == "assistant" else None,
            robot_name=self.robot_name if role == "assistant" else None,
            user_id=None, # Assuming assistant role primarily
            user_name=None, # Assuming assistant role primarily
            content=content,
            role=role,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata,
        )

    # --- End Payload Helpers ---

    def _init_system_prompt(self) -> None:
        """Initialize the base system prompt. Override in child classes."""
        self.system_prompt = SystemMessage(content="I am a base AI robot.")
        if self.prompt_manager:
            self.prompt_manager.set_system_prompt(self.system_prompt)

    async def update_system_prompt(self) -> None:
        """Update the system prompt. Override in child classes."""
        if not self.system_prompt:
            self._init_system_prompt()
        elif self.prompt_manager and not self.prompt_manager.system_messages:
            self.prompt_manager.set_system_prompt(self.system_prompt)

    @abstractmethod
    def stop_condition(self) -> bool:
        """
        This has to be implemented by the child class.
        It should return a boolean value indicating whether the interaction should stop.
        Example:
        def stop_condition(self) -> bool:
            return len(self.conversation_history) >= 5
        DEFAULTS TO self.finished, so you should override this if you want custom behaviour
        """
        return self.finished

    async def stop(self) -> None:
        """Stop the robot's operation."""
        self.finished = True
        # Clear this robot from the message handler
        self.message_handler.clear_current_robot()
        # Only close the message handler if we own it
        if self.owns_message_handler and isinstance(
            self.message_handler, ConsoleMessageHandler
        ):
            try:
                await self.message_handler.websocket.close()
            except Exception as e:
                logger.warning(f"Error closing websocket: {e}")

    async def connect(self) -> None:
        """Get initial input through the message handler
        - You don't need to define this if you're not using IO chat like code"""

    @abstractmethod
    async def prepare(self) -> None:
        """
        Prepare the robot for interaction. Override this method to:
        - Set up the prompt
        - Initialize any required data
        - Configure any pre-processing steps
        """
        pass

    @abstractmethod
    async def process(self) -> None:
        """
        Process the interaction. Override this method to:
        - Handle the AI response
        - Process any function calls
        - Transform data as needed
        """
        pass

    async def load(self, input_data: InputType) -> None:
        """
        Load input data into the robot.

        This method provides explicit type checking for input data through the InputType generic.
        Child classes should override this method to handle their specific input type.

        Args:
            input_data: The input data matching the robot's InputType generic parameter
        """
        self.input_data = input_data

    async def output(self) -> OutputType:
        """
        Get the robot's output data.

        This method provides explicit type checking for output data through the OutputType generic.
        Child classes should override this method to format their specific output type.

        Returns:
            The output data matching the robot's OutputType generic parameter
        """
        return self.output_data

    async def finalize(self) -> None:
        """
        Finalize the interaction. Override this method to:
        - Clean up resources
        - Save results
        - Prepare for next iteration
        - Robots do not need to define this, but iis called after process()
        """
        pass

    async def _interactive_generate(self, *args, **kwargs):
        """Interactive version of generate_response for testing - simulates AI responses"""
        menu = Table(show_header=True, header_style="bold magenta")
        menu.add_column("Option")
        menu.add_column("Description")
        menu.add_row("1", "Send a text message")
        menu.add_row("2", "Call a function")
        menu.add_row("3", "Show Current Context")
        menu.add_row("4", "Exit")

        console = Console()
        console.print("\n=== HUMAN AI MODE ===", style="bold green")
        console.print(menu)
        choice = Prompt.ask(
            "What would you like the AI to do?", choices=["1", "2", "3", "4"]
        )

        if choice == "1":
            message = Prompt.ask("Enter your message")

            async def message_generator():
                yield ChatCompletionChunk(
                    id="test",
                    choices=[
                        Choice(
                            delta=ChoiceDelta(content=message),
                            finish_reason=None,
                            index=0,
                        )
                    ],
                    created=int(time.time()),
                    model="test",
                    object="chat.completion.chunk",
                )
                yield ChatCompletionChunk(
                    id="test",
                    choices=[
                        Choice(delta=ChoiceDelta(), finish_reason="stop", index=0)
                    ],
                    created=int(time.time()),
                    model="test",
                    object="chat.completion.chunk",
                )

            return message_generator()

        elif choice == "2":
            functions = args[1] if len(args) > 1 else {}
            if not functions:
                console.print("[yellow]No functions available[/yellow]")

                # Return empty generator instead of None
                async def empty_generator():
                    yield ChatCompletionChunk(
                        id="test",
                        choices=[
                            Choice(delta=ChoiceDelta(), finish_reason="stop", index=0)
                        ],
                        created=int(time.time()),
                        model="test",
                        object="chat.completion.chunk",
                    )

                return empty_generator()

            # Show available functions
            for i, name in enumerate(functions.keys(), 1):
                console.print(f"{i}. {name}")

            while True:  # Keep asking until we get valid input
                try:
                    func_choice = IntPrompt.ask(
                        "Select function to call (number)",
                        choices=[str(i) for i in range(1, len(functions) + 1)],
                    )
                    break
                except ValueError:
                    console.print("[red]Please enter a valid number[/red]")

            function_name = list(functions.keys())[func_choice - 1]
            function = list(functions.values())[func_choice - 1]

            params = {}
            desc = function.description()

            if "parameters" in desc.get("function", {}):
                param_info = desc["function"]["parameters"]
                required_params = set(param_info.get("required", []))
                properties = param_info.get("properties", {})

                for name, details in properties.items():
                    is_required = name in required_params
                    while True:
                        value = Prompt.ask(
                            f"{name} ({details.get('type', 'string')})"
                            + (" [required]" if is_required else ""),
                            default="" if is_required else None,
                        )

                        if not value and is_required:
                            console.print(
                                f"[red]'{name}' is required. Please enter a value.[/red]"
                            )
                            continue

                        if value or is_required:
                            params[name] = value
                        break

            async def function_generator():
                yield ChatCompletionChunk(
                    id="test",
                    choices=[
                        Choice(
                            delta=ChoiceDelta(
                                tool_calls=[
                                    ChoiceDeltaToolCall(
                                        index=0,
                                        function=ChoiceDeltaToolCallFunction(
                                            name=function_name,
                                            arguments="",
                                        ),
                                    )
                                ]
                            ),
                            finish_reason=None,
                            index=0,
                        )
                    ],
                    created=int(time.time()),
                    model="test",
                    object="chat.completion.chunk",
                )
                yield ChatCompletionChunk(
                    id="test",
                    choices=[
                        Choice(
                            delta=ChoiceDelta(
                                tool_calls=[
                                    ChoiceDeltaToolCall(
                                        index=0,
                                        function=ChoiceDeltaToolCallFunction(
                                            name=None,
                                            arguments=json.dumps(params),
                                        ),
                                    )
                                ]
                            ),
                            finish_reason=None,
                            index=0,
                        )
                    ],
                    created=int(time.time()),
                    model="test",
                    object="chat.completion.chunk",
                )
                yield ChatCompletionChunk(
                    id="test",
                    choices=[
                        Choice(delta=ChoiceDelta(), finish_reason="tool_calls", index=0)
                    ],
                    created=int(time.time()),
                    model="test",
                    object="chat.completion.chunk",
                )

            return function_generator()

        elif choice == "3":
            messages = self.prompt_manager.all_messages
            context = "\n\n".join([f"[{m.role}]: {m.content}" for m in messages])
            console.print(Panel(context, title="Current Context", style="blue"))
            return await self._interactive_generate(*args, **kwargs)

        else:  # Exit
            self.finished = True

            # Return empty generator instead of None
            async def empty_generator():
                yield ChatCompletionChunk(
                    id="test",
                    choices=[
                        Choice(delta=ChoiceDelta(), finish_reason="stop", index=0)
                    ],
                    created=int(time.time()),
                    model="test",
                    object="chat.completion.chunk",
                )

            return empty_generator()

    async def interact(self, test: bool = False) -> OutputType:
        """Main interaction loop. Assumes setup_interaction was called first."""
        if not self.interaction_id:
            await self.setup_interaction(str(uuid_pkg.uuid4()), None)
            logger.warning(
                "interact() called without prior setup_interaction(). Generated fallback interaction ID."
            )

        # --- Testing mode setup --- (Remains the same)
        if test:
            self.testing = True
            original_handler = self.message_handler
            original_generate = self.ai_class.generate_response
            if not isinstance(self.message_handler, ConsoleMessageHandler):
                self.message_handler = ConsoleMessageHandler()
                # Ensure handler knows about the robot context
                self.message_handler.set_current_robot(self)
                self.ai_class.message_handler = self.message_handler
            self.ai_class.generate_response = self._interactive_generate
        # --------------------------

        try:
            await (
                self.connect()
            )  # Should this also be part of setup? Maybe not if handler specific.
            await self.prepare()

            if not self.prompt_manager.all_messages:
                raise ValueError("Prompt not set - you must set a prompt in prepare()")

            if not self.skip_call:
                await self.generate_ai_response()  # Uses new handler methods indirectly

            # Output logging now handled by process or specific handler calls

            # Process uses updated handler methods
            await self.process()

            if await self.stop_condition():
                result = await self.finalize()
                # --- Testing mode teardown --- (Remains the same)
                if test:
                    self.message_handler = original_handler
                    self.ai_class.generate_response = original_generate
                # ---------------------------
                return result

            # Recursive call continues the SAME interaction/run ID by default
            return await self.interact(test=test)

        except (WebSocketDisconnect, ConnectionClosedOK, ConnectionClosedError) as e:
            # Log error using self.log which uses handler.update_status or handler.send_error
            error_payload = self._create_error_payload(message = f"WebSocket disconnected: {e}" )
            await self.message_handler.send_error(error_payload)
            raise
        except Exception as e:
            # Log error and send explicit error via handler
            error_msg = f"Exception in interact loop: {e}"
            logger.exception(error_msg, exc_info=True)  # Log full traceback
            # await self.log(error_msg, level="error") # Log method sends status/error
            if self.message_handler:
                try:
                    error_payload = ErrorPayload(
                        interaction_id=self.interaction_id,
                        message=f"Robot execution failed: {e}",
                        code="ROBOT_EXECUTION_ERROR",
                        context={"traceback": format_exc()},  # Add traceback
                    )
                    await self.message_handler.send_error(error_payload)
                except Exception as handler_err:
                    logger.error(
                        f"Failed to send error via message handler during interact exception: {handler_err}"
                    )
            raise
        finally:
            # --- Testing mode teardown --- (Remains the same, ensure it runs)
            if test:
                self.message_handler = original_handler
                self.ai_class.generate_response = original_generate
            # ---------------------------

    async def generate_ai_response(self):
        """Get response from AI using available functions from toolsets"""
        # If something has restricted the functions, use those
        if self.restricted_functions:
            functions_to_pass_to_ai = self.restricted_functions
        else:
            # Otherwise, use the available functions
            functions_to_pass_to_ai = self.available_functions

        # Call BaseAI's generate_response
        response_data = await self.ai_class.generate_response(
            self.prompt_manager.all_messages,
            functions_to_pass_to_ai,
            self.force_function,
        )

        if self.stream:
            # response_data is AsyncGenerator[ChatCompletionChunk, None]
            self.output_data = response_data  # Store generator
        else:
            # response_data is Tuple[str, Optional[List[Any]]]
            output_content, function_calls = response_data
            # Store RobaiChatMessagePayload internally, handler methods will use it later
            self.output_data = RobaiChatMessagePayload(
                role="assistant",
                content=output_content or '',
                name=self.robot_name,  # Use robot's name
                robot_id=self.robot_id,     
                interaction_id=self.interaction_id,
                chat_id=self.chat_id,
            )
            self.pending_function_calls = function_calls or []  # Ensure list

        # IMPORTANT - reset restricted functions *after* getting the response
        self.restricted_functions = {}

    # UTILITY FUNCTIONS FOR CHILD CLASSES
    async def _handle_non_streaming_response(self) -> None:
        """Process a non-streaming AI response, ensure self.output_data is RobaiChatMessagePayload, and use handler."""
        final_ai_message: Optional[RobaiChatMessagePayload] = None

        # 1. Check the initial output_data type
        if isinstance(self.output_data, str):
            # If it's a string, wrap it in an RobaiChatMessagePayload
            logger.debug("Wrapping string output_data in RobaiChatMessagePayload for non-streaming.")
            final_ai_message = RobaiChatMessagePayload(
                role="assistant",
                content=self.output_data,
                name=self.robot_name,
                robot_id=self.robot_id,
                interaction_id=self.interaction_id,
                chat_id=self.chat_id,
            )
            # Update self.output_data to the RobaiChatMessagePayload object
            self.output_data = final_ai_message

        elif isinstance(self.output_data, RobaiChatMessagePayload):
            # If it's already an RobaiChatMessagePayload, use it directly
            logger.debug("Using existing RobaiChatMessagePayload output_data for non-streaming.")
            final_ai_message = self.output_data
        else:
            # If it's neither string nor RobaiChatMessagePayload, log error and try to send error payload
            logger.error(
                f"_handle_non_streaming_response called with unexpected output_data type: {type(self.output_data)}. Cannot process."
            )
            if self.message_handler:
                try:
                    # Use helper to create error payload
                    error_payload = self._create_error_payload(
                        message="Internal error: Invalid non-streaming output format generated by AI.",
                        code="INTERNAL_OUTPUT_ERROR"
                    )
                    await self.message_handler.send_error(payload=error_payload) # Pass created payload
                except Exception as handler_err:
                    logger.error(
                        f"Failed to send output format error via message handler: {handler_err}"
                    )
            # Set self.output_data to None to signify failure
            self.output_data = None
            return  # Exit if format is invalid

        # 2. Ensure we have a valid RobaiChatMessagePayload and content
        if not final_ai_message:
            logger.error(
                "Failed to create or retrieve RobaiChatMessagePayload in _handle_non_streaming_response."
            )
            self.output_data = None  # Ensure output_data reflects failure
            return

        content_to_send = final_ai_message.content
        if content_to_send is None:
            logger.warning(
                f"Non-streaming response content is None for run {self.interaction_id}. Sending empty message."
            )
            content_to_send = ""  # Use empty string
            # Update the message object itself if needed for history
            final_ai_message.content = content_to_send

        # 3. Send the full message via the handler using helper
        payload = self._create_full_message_payload(
            content=content_to_send,
            role="assistant",
            metadata=final_ai_message.metadata # Pass metadata if any
        )
        try:
            if self.message_handler:
                await self.message_handler.send_full_message(payload=payload) # Pass created payload
                # Add the final RobaiChatMessagePayload to internal history
                self.add_message(final_ai_message)
                self.add_action_to_history("✅ Full Message Sent")
            else:
                logger.warning(
                    "No message handler available to send non-streaming response."
                )
                # Still add to history even if not sent?
                self.add_message(final_ai_message)
                self.add_action_to_history("✅ Message Generated (No Handler)")

        except Exception as e:
            logger.exception(f"Error sending non-streaming message via handler: {e}")
            self.add_action_to_history("❌ Error Sending Message", details=str(e))
            # Error already logged, output_data remains the RobaiChatMessagePayload but sending failed
            if self.message_handler:
                try:
                    # Use helper to create error payload
                    error_payload = self._create_error_payload(
                        message=f"Failed to send generated message via handler: {e}",
                        code="HANDLER_SEND_ERROR"
                    )
                    await self.message_handler.send_error(payload=error_payload) # Pass created payload
                except Exception:
                    pass  # Avoid error loops

    async def _process_chunk(self, chunk: ChatCompletionChunk, message_id: str) -> None:
        """Process individual chunks from the AI response stream."""
        delta = chunk.choices[0].delta

        # Handle content chunks
        if delta.content:
            self._accumulated_message += delta.content
            await self.message_handler.stream_chunk(
                StreamChunkPayload(
                    interaction_id=self.interaction_id,
                    content=delta.content,
                    message_id=message_id,
                )
            )

        # Handle tool call chunks
        if delta.tool_calls:
            await self._handle_tool_call_chunk(delta.tool_calls[0].function)
            # If we get a finish_reason, finalize the current function
            if chunk.choices[0].finish_reason:
                await self._finalize_current_function()

    async def _handle_tool_call_chunk(
        self, function_delta: ChoiceDeltaToolCallFunction
    ) -> None:
        """Process function call chunks from the AI response."""
        if not self._current_function:
            self._current_function = {"name": "", "arguments": ""}
            if function_delta.name:
                # Use helper to create status payload
                status_payload = self._create_status_payload(status=f"Calling {function_delta.name}")
                await self.message_handler.update_status(payload=status_payload)

        # Accumulate function name
        if (
            function_delta.name
            and function_delta.name not in self._current_function["name"]
        ):
            self._current_function["name"] += function_delta.name

        # Accumulate arguments
        if function_delta.arguments:
            self._current_function["arguments"] += function_delta.arguments

    async def _finalize_current_function(self) -> None:
        """Complete the current function call if one exists. Sets self.pending_function_calls"""
        if self._current_function:
            function_call = ChoiceDeltaToolCallFunction(
                name=self._current_function["name"] or None,
                arguments=self._current_function["arguments"],
            )
            self.pending_function_calls.append(function_call)
            self._current_function = None

    async def _execute_function_calls(self) -> None:
        """Execute functions the AI requested, results go directly to self.function_results"""
        if not self.pending_function_calls:
            return

        executed_call_ids = []  # Keep track of executed calls for final results message
        for tool_call in self.pending_function_calls:
            if isinstance(tool_call, ChoiceDeltaToolCallFunction):
                executed_call_ids.append(tool_call.name)
                function_name = tool_call.name
                arguments = tool_call.arguments
            elif isinstance(tool_call, ChatCompletionMessageToolCall):
                executed_call_ids.append(tool_call.function.name)
                function_name = tool_call.function.name
                arguments = tool_call.function.arguments

            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                error_message = f"Error parsing arguments for {function_name}: {e}, {tool_call['function']['arguments']}"
                logger.error(error_message)
                self.function_results.add_error(error_message)
                raise  # Re-raise to make the error visible

            # Check against all_functions since restricted functions might not be in available_functions
            if function_name in self.all_functions:
                try:
                    method = getattr(self, function_name)
                    await method(**parsed_arguments)
                except Exception as e:
                    error_message = (
                        f"Error in function '{function_name}' {str(e)}"
                    )
                    logger.exception(
                        f"Function execution failed: {function_name}",
                        exc_info=True,
                    )
                    self.function_results.add_error(
                        error_message
                    )  # Add simplified error for context
                    # Send detailed error via handler
                    # --- Combine into a single f-string ---
                    detailed_error = f"Function Call Error: {function_name}\nArguments: {json.dumps(parsed_arguments, indent=2)}\nError: {str(e)}\nTraceback:\n{traceback.format_exc()}"
                    # -------------------------------------
                    # Use helper to create error payload
                    error_payload = self._create_error_payload(
                        message=detailed_error, 
                        code="FUNCTION_EXECUTION_ERROR",
                        context={"function_name": function_name}
                    )
                    try:
                        await self.message_handler.send_error(payload=error_payload) # Pass created payload
                    except Exception as handler_err:
                        logger.error(
                            f"Failed to send function execution error via handler: {handler_err}"
                        )
                    # Continue loop even if one function fails
            else:
                error_message = f"Tool {function_name} not found"
                self.function_results.add_error(error_message)
                raise ValueError(error_message)

        # Use helper to create status payload
        final_status_payload = self._create_status_payload(
            status=f"**FUNCTIONS CALLED!** RESULTS: {self.function_results.__str__()}"
        )
        await self.message_handler.update_status(payload=final_status_payload) # Pass created payload
        self.function_results = MarkdownFunctionResults()  # Reset for next round

    async def _handle_streaming_response(self) -> None:
        """Process a streaming AI response using new handler methods."""
        try:
            await self._stream_response_and_gather_functions()
            if self.pending_function_calls:
                await self._execute_function_calls()
                # Add function results to context *after* execution
                await (
                    self._add_function_results_to_context()
                )  # Add results for next AI turn
        except Exception as e:
            # Error handling is now primarily within _stream_response_and_gather_functions
            # and _execute_function_calls. This block is a final catch-all.
            logger.exception(
                f"Unhandled exception during _handle_streaming_response for run {self.interaction_id}: {e}"
            )
            if self.message_handler:
                try:
                    # Use helper to create error payload
                    error_payload = self._create_error_payload(
                        message=f"Unexpected error handling stream: {e}",
                        code="STREAM_HANDLING_ERROR",
                        context={"traceback": format_exc()}
                    )
                    await self.message_handler.send_error(payload=error_payload) # Pass created payload
                except Exception as handler_err:
                    logger.error(
                        f"Failed to send final stream handling error via handler: {handler_err}"
                    )
            # Depending on design, might want to ensure stream is marked ended

    async def _stream_response_and_gather_functions(self) -> None:
        """Process stream, accumulate message/functions, use new handler methods."""
        self._accumulated_message = ""
        self._current_function = None
        self.pending_function_calls: List[Union[ChatCompletionMessageToolCall, ChoiceDeltaToolCallFunction]] = []
        internal_message_to_add: Optional[RobaiChatMessagePayload] = (
            None  # Store message to add to history at the end
        )

        try:
            # 1. Send stream_start using helper
            start_payload = self._create_stream_start_payload()
            await self.message_handler.stream_start(payload=start_payload) # Pass created payload

            # 2. Process chunks
            if not isinstance(self.output_data, openai.AsyncStream):
                logger.error(
                    f"Stream processing error: output_data is not an AsyncGenerator (type: {type(self.output_data)}) for run {self.interaction_id}"
                )
                raise TypeError("Cannot process stream, invalid output data.")

            async for chunk in self.output_data:
                chunk: ChatCompletionChunk
                await self._process_chunk(chunk)  # Pass only chunk
                if chunk.choices and chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
                    logger.debug(
                        f"Stream finish reason for run {self.interaction_id}: {finish_reason}"
                    )
                    if finish_reason == "tool_calls":
                        await self._finalize_current_function()
                    break  # Exit loop on any finish reason

            # Always finalize any pending function call after loop
            await self._finalize_current_function()
            logger.debug(
                f"Finished processing stream chunks for run {self.interaction_id}"
            )

            # 3. Prepare internal message representation and send final message
            if self._accumulated_message or self.pending_function_calls:
                internal_message_to_add = RobaiChatMessagePayload(
                    role="assistant",
                    content=self._accumulated_message or "",
                    name=self.robot_name,
                    robot_id=self.robot_id,
                    interaction_id=self.interaction_id,
                    chat_id=self.chat_id,
                )
                # Send final message payload using helper
                final_message_payload = self._create_full_message_payload(
                    content=self._accumulated_message,
                    role="assistant"
                    # metadata could be added here if needed
                )
                await self.message_handler.send_full_message(payload=final_message_payload)

                # Then send stream_end using helper
                end_payload = self._create_stream_end_payload(final_content=self._accumulated_message)
                await self.message_handler.stream_end(payload=end_payload)
                logger.debug(f"Stream ended successfully for run {self.interaction_id}")

                # Add to history after successful sends
                self.add_message(internal_message_to_add)
                self.output_data = internal_message_to_add
                logger.debug(
                    f"Added final stream message to internal history for run {self.interaction_id}"
                )
            else:
                # Even with no content, send stream end using helper
                end_payload = self._create_stream_end_payload(final_content="")
                await self.message_handler.stream_end(payload=end_payload) # Pass created payload
                logger.debug(f"Stream ended (no content) for run {self.interaction_id}")

        except Exception as e:
            logger.exception(f"Error in _stream_response_and_gather_functions: {e}")
            # Use helper to create error payload
            error_payload = self._create_error_payload(
                message=f"Error processing stream: {str(e)}",
                code="STREAM_PROCESSING_ERROR",
                context={"robot_id": self.robot_id} # Keep specific context if needed
            )
            await self.message_handler.send_error(payload=error_payload) # Pass created payload
            raise

    async def _process_chunk(self, chunk: ChatCompletionChunk) -> None:
        """Process individual chunk, send via handler, accumulate message/functions."""
        if not chunk.choices:
            return
        delta = chunk.choices[0].delta

        # Handle content chunks
        if delta.content:
            self._accumulated_message += delta.content
            # Use helper to create chunk payload
            # Note: message_id is not currently used by the helper but passed for potential future use
            chunk_payload = self._create_stream_chunk_payload(content=delta.content)
            await self.message_handler.stream_chunk(payload=chunk_payload) # Pass created payload

        # Handle tool call chunks
        if delta.tool_calls:
            tool_call_delta = delta.tool_calls[0]
            if tool_call_delta.function:
                await self._handle_tool_call_chunk(tool_call_delta.function)

    def set_system_prompt(self, prompt: SystemMessage) -> None:
        """Set the system prompt, replacing any existing ones."""
        self.system_prompt = prompt
        self.prompt_manager.clear_all()
        self.prompt_manager.set_system_prompt(prompt)

    def add_message(self, message: Union[RobaiChatMessagePayload]) -> None:
        """Add a message to the chat window."""
        self.prompt_manager.add_message(message)

    async def _add_function_results_to_context(self) -> None:
        """Add function results as regular messages that can be trimmed by the chat window."""
        if str(self.function_results):
            # Add as a regular message instead of system message so it can be trimmed
            self.prompt_manager.add_message(
                RobaiChatMessagePayload(
                    role="assistant",
                    robot="function_bot",
                    content=f"**FUNCTIONS CALLED!** RESULTS: {self.function_results.__str__()}",
                    interaction_id=self.interaction_id,
                )
            )
            self.function_results = MarkdownFunctionResults()  # Reset for next round

    def enable_context_debugging(self):
        """Enable printing of the full context at each iteration"""
        self.debug_context = True

    def disable_context_debugging(self):
        """Disable context debugging"""
        self.debug_context = False

    def get_current_context(self) -> str:
        """Get the current context as a formatted string"""
        context = []
        for msg in self.prompt_manager.all_messages:
            context.append(f"[{msg.role.upper()}]:\n{msg.content}\n")
        return "\n".join(context)

    def restrict_functions(self, allowed_functions: List[str]) -> None:
        """Set which functions will be available in the NEXT AI call."""
        self.restricted_functions = {
            name: self.all_functions[name]
            for name in allowed_functions
            if name in self.all_functions
        }

    def _get_warnings(self) -> List[str]:
        """Generate warning messages based on behavior patterns"""
        warnings = []

        # Only analyze if we have history
        if not self.action_history:
            return warnings

        # Analyze last 5 actions for patterns
        recent_actions = self.action_history[-5:]

        # Count action types (using first word/emoji as type)
        action_types = {}
        for timestamp, action in recent_actions:
            action_type = action.split()[0]  # Get first word/emoji
            action_types[action_type] = action_types.get(action_type, 0) + 1

        # Check for repetitive patterns
        if any(count >= 3 for count in action_types.values()):
            repeated_actions = [
                action for action, count in action_types.items() if count >= 3
            ]
            warnings.append(
                f"⚠️ You're repeating similar actions ({', '.join(repeated_actions)}). "
                "Try varying your approach!"
            )

        # Check for multiple errors
        error_actions = sum(1 for (_, action) in recent_actions if "❌" in action)
        if error_actions >= 2:
            warnings.append(
                "⚠️ Multiple recent errors detected - consider trying a different strategy"
            )

        # Check for action frequency
        for action_type, count in self.action_counts.items():
            if count > self.max_history_items * 0.4:  # If any action is >40% of history
                warnings.append(
                    f"⚠️ Heavy reliance on '{action_type}' detected. "
                    "Try using other available functions."
                )

        # Check time patterns
        if len(recent_actions) >= 2:
            timestamps = [ts for ts, _ in recent_actions]
            time_diffs = [
                (timestamps[i + 1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps) - 1)
            ]
            if all(diff < 2 for diff in time_diffs):  # If all actions within 2 seconds
                warnings.append(
                    "⚠️ Actions happening too quickly. Take time to analyze results."
                )

        return warnings

    def add_action_to_history(self, action: str, details: str = None) -> None:
        """
        Add an action to history with timestamp and optional details.

        Args:
            action: The main action description (should start with an emoji for categorization)
            details: Optional additional details about the action
        """
        timestamp = datetime.now()

        # Track action type frequency
        action_type = action.split()[0]  # Get first word/emoji
        self.action_counts[action_type] = self.action_counts.get(action_type, 0) + 1

        # Format the action with details if provided
        formatted_action = action
        if details:
            formatted_action += f": {details}"

        # Track errors
        if "❌" in action:
            self.error_count += 1

        # Add to history
        self.action_history.append((timestamp, formatted_action))

        # Trim history if needed
        if len(self.action_history) > self.max_history_items:
            self.action_history = self.action_history[-self.max_history_items :]

        # Reset error count when history is trimmed
        if len(self.action_history) == self.max_history_items:
            self.error_count = sum(
                1 for _, action in self.action_history if "❌" in action
            )

    def get_action_history(self) -> str:
        """
        Get action history formatted with human-readable relative timestamps.
        Returns a string with each action on a new line.
        """
        now = datetime.now()
        formatted_history = []

        for timestamp, action in self.action_history:
            # Calculate time difference
            delta = now - timestamp

            # Format time difference in a human-readable way
            if delta.total_seconds() < 60:
                time_ago = f"{int(delta.total_seconds())}s ago"
            elif delta.total_seconds() < 3600:
                time_ago = f"{int(delta.total_seconds() / 60)}m ago"
            elif delta.total_seconds() < 86400:
                time_ago = f"{int(delta.total_seconds() / 3600)}h ago"
            else:
                time_ago = f"{int(delta.total_seconds() / 86400)}d ago"

            # Format the line with the relative timestamp
            formatted_history.append(f"[{time_ago}] {action}")

        # Join all lines with newlines
        return "\n".join(formatted_history)

    async def _handle_function_response(self, context_message: str) -> None:
        """Handle function results by getting AI to analyze them and respond.

        Args:
            context_message: The message to add to context for AI to analyze

        This method:
        1. Adds the context message to prompt manager
        2. Generates an AI response analyzing the results
        3. Handles the streaming response
        4. Saves the response to history if using WebSocket handler
        """
        # Add context for AI to analyze
        self.prompt_manager.add_message(SystemMessage(content=context_message))

        # Get AI's analysis
        await self.generate_ai_response()

        # Stream the response
        await self._handle_streaming_response()

        # Save to history if we're using WebSocket handler
        await self.message_handler.save_to_history(self.output_data)

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.owns_message_handler and isinstance(
            self.message_handler, ConsoleMessageHandler
        ):
            try:
                await self.message_handler.websocket.close()
            except Exception as e:
                logger.warning(f"Error closing websocket during cleanup: {e}")

    async def setup_interaction(
        self, interaction_id: str, chat_id: Optional[UUID] = None
    ):
        """Sets up the context for a specific interaction run."""
        if not interaction_id:
            logger.warning("Robot interaction started without an interaction_id!")
            self.interaction_id = str(uuid_pkg.uuid4())  # Generate fallback
        else:
            self.interaction_id = interaction_id
        self.chat_id = chat_id
        logger.debug(
            f"Robot {self.robot_id} interaction setup: InteractionID={self.interaction_id}, ChatID={self.chat_id}"
        )
        # Reset state variables for the new interaction?
        self.finished = False
        self.skip_call = False
        # Reset prompt manager? Depends on desired behavior (persistent vs. per-interaction context)
        # self.prompt_manager.clear_messages() # Uncomment if context should reset
