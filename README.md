# RobAI Framework

**RobAI** is a lightweight, Pythonic framework for building AI-powered robots. Created by `philmade`, it streamlines integration of AI models (e.g., OpenAI's GPT) with a memory-driven workflow: prepare input, process AI responses, and handle function calls—all in a clean, extensible way. Ideal for chatbots, interactive tools, or AI-driven apps with minimal setup hassle.

- **Version**: 0.1.0
- **Source**: [github.com/philmade/robai](https://github.com/philmade/robai)
- **License**: MIT

## Features

- **Robot Classes**: Extend `BaseRobot` or `ChatRobot` for custom AI logic.
- **Memory Handling**: `PromptManager` manages chat history with token/message limits.
- **Function Calling**: Define AI-callable methods with `@robot_function` and Pydantic schemas.
- **Streaming**: Real-time AI responses via OpenAI's streaming API.
- **Flexible I/O**: Console, WebSocket, or custom handlers for input/output.
- **Pythonic**: Simple APIs, type hints, and decorators keep it intuitive.

## Installation

Install `robai` from GitHub:

```bash
pip install git+https://github.com/philmade/robai.git
```

### Dependencies (auto-installed):

- pydantic>=1.10.7
- openai>=1.0.0
- tiktoken>=0.5.0
- loguru>=0.7.0
- rich>=13.0.0
- Others in setup.py

Requires Python 3.11+.

## Quick Start

Based on example.py, here's how to create a chat bot with function-calling capabilities:

```python
from robai.schemas import ChatMessage, SystemMessage
from robai.chat import ChatRobot
from robai.protocols import ConsoleMessageHandler
from robai.func_tools import robot_function
from pydantic import BaseModel, Field

# Pydantic model for input
class HelloWorldInput(BaseModel):
    message: str = Field(..., description="A message to print")

# Define the bot
class TestBot(ChatRobot):
    def __init__(self):
        super().__init__(message_handler=ConsoleMessageHandler())
        self.system_prompt = SystemMessage(
            content="You're a helpful assistant. Use functions to respond when possible."
        )

    @robot_function()
    async def hello_world(self, input_data: HelloWorldInput) -> None:
        """Prints a message to the console"""
        print(input_data.message)
        self.function_results.append("hello_world", input_data.message)

    async def prepare(self):
        message = await self.message_handler.wait_for_input()
        self.prompt_manager.set_system_prompt(self.system_prompt)
        self.prompt_manager.add_message(message)

    async def process(self):
        await self._handle_streaming_response()
        if self.pending_function_calls:
            await self._add_function_results_to_context()
            self.prompt_manager.add_message(
                SystemMessage(content="Respond to the function results.")
            )
            await self.generate_ai_response()
            await self._handle_streaming_response()

    async def stop_condition(self):
        return self.finished

# Run it
import asyncio
async def main():
    bot = TestBot()
    await bot.interact()
asyncio.run(main())
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-key-here"
```

Run it, type "Say hi", and the AI might call hello_world to print "hi" while explaining the action.

## Directory Structure

```
robai/
├── __init__.py     # Core exports
├── base.py         # BaseRobot, BaseAI, PromptManager
├── chat.py         # ChatRobot for chat-based bots
├── errors.py       # Custom exceptions
├── example.py      # Demo bot with functions
├── func_tools.py   # Function decorators
├── protocols.py    # Message handlers
├── schemas.py      # Pydantic models
├── tools.py        # Tool management
├── utility.py      # Helpers and logging
└── setup.py        # Package setup
```

## Key Components

### TestBot (from example.py)
- **Init**: Sets a system prompt and uses ConsoleMessageHandler
- **Functions**:
  - hello_world: Prints a message
  - add_numbers: Adds two numbers
  - process_array: Joins a list of strings
  - process_complex: Handles nested objects
  - exit_conversation: Stops the bot
- **Flow**: Takes input, calls functions if prompted, and responds with results

### BaseRobot
- prepare(): Sets up prompts and input
- process(): Handles AI responses and function calls
- stop_condition(): Defines when to stop (e.g., after one response)

### PromptManager
- Manages chat history with max_messages (50) and max_tokens (4000)

### MessageHandler
- ConsoleMessageHandler: Terminal I/O with rich formatting

## Usage Example

Try the full TestBot from example.py:

```python
from robai.schemas import ChatMessage, SystemMessage, AIMessage
from robai.chat import ChatRobot
from robai.protocols import ConsoleMessageHandler
from robai.func_tools import robot_function
from pydantic import BaseModel, Field

class AddNumbersInput(BaseModel):
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")

class TestBot(ChatRobot):
    def __init__(self):
        super().__init__(message_handler=ConsoleMessageHandler())
        self.system_prompt = SystemMessage(
            content="You're a helpful assistant. Use functions when asked."
        )

    @robot_function()
    async def add_numbers(self, data: AddNumbersInput) -> None:
        """Add two numbers"""
        result = data.a + data.b
        self.function_results.append("add_numbers", f"Result: {result}")

    async def prepare(self):
        message = await self.message_handler.wait_for_input()
        self.prompt_manager.set_system_prompt(self.system_prompt)
        self.prompt_manager.add_message(message)

    async def process(self):
        await self._handle_streaming_response()
        if self.pending_function_calls:
            await self._add_function_results_to_context()
            self.prompt_manager.add_message(
                SystemMessage(content="Tell the user what happened.")
            )
            await self.generate_ai_response()
            await self._handle_streaming_response()

    async def stop_condition(self):
        return self.finished

import asyncio
async def main():
    bot = TestBot()
    await bot.interact()
asyncio.run(main())
```

Type "Add 3 and 5", and it'll call add_numbers, print "Result: 8", and explain the process.

## Usage Tips

### Functions
Use `@robot_function` with Pydantic models for structured input:

```python
@robot_function()
async def say_hi(self, input: HelloWorldInput):
    return f"Hi, {input.message}!"
```

### Additional Features
- **Streaming**: Enable with stream=True in ChatRobot
- **Exit**: Say "exit" to trigger exit_conversation

## Requirements

- OpenAI API key (OPENAI_API_KEY)
- Python 3.11+
- Dependencies from setup.py

## Contributing

Fork [github.com/philmade/robai](https://github.com/philmade/robai), make changes, and submit a PR. Issues welcome!

## Why RobAI?

It's simple, extensible, and avoids build nightmares. Pair it with a GUI (e.g., BeeWare's Toga) for a slick AI app in no time!