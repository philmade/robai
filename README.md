RobAI Framework
RobAI is a lightweight, Pythonic framework for building AI-powered robots. It simplifies integrating AI models (like OpenAI’s GPT) into your applications with a clean, memory-driven workflow: pre-call processing, AI interaction, and post-call handling. Designed by philmade, it’s perfect for creating conversational agents, function-calling bots, or interactive tools—all with minimal fuss.

Version: 0.1.0
Source: github.com/philmade/robai
License: MIT
Features
SimpleRobot Base: Extend BaseRobot or ChatRobot to craft custom AI bots with built-in message management.
Memory Management: PromptManager handles chat history with token and message limits, trimming old messages automatically.
Function Calling: Use @robot_function to define AI-callable methods, with Pydantic-powered schema validation.
Streaming Support: Real-time AI responses via OpenAI’s streaming API.
Cross-Platform I/O: Console, WebSocket, or custom message handlers for flexible input/output.
Pythonic Design: Intuitive APIs, type hints, and decorators make it feel like pure Python.
Installation
Install robai directly from GitHub:

bash

Collapse

Wrap

Copy
pip install git+https://github.com/philmade/robai.git
Required dependencies (auto-installed):

pydantic>=1.10.7
openai>=1.0.0
tiktoken>=0.5.0
loguru>=0.7.0
rich>=13.0.0
Plus a few others (see setup.py).
Python 3.11+ is required.

Quick Start
Here’s a minimal example of a chat robot that echoes your input with an AI twist:

python

Collapse

Wrap

Copy
from robai.base import BaseRobot
from robai.schemas import ChatMessage, SystemMessage

class EchoBot(BaseRobot):
    def _init_system_prompt(self):
        self.system_prompt = SystemMessage(content="Echo back what the user says, but make it fun!")

    async def prepare(self):
        self.prompt_manager.add_message(self.input_data)

    async def process(self):
        await self.generate_ai_response()
        await self._handle_streaming_response()
        self.output_data = self.output_data  # Streamed response is set here

    async def stop_condition(self):
        return self.finished

# Run it
async def main():
    bot = EchoBot()
    await bot.load(ChatMessage(content="Hello!"))
    await bot.interact()
    print(bot.output_data.content)

import asyncio
asyncio.run(main())
Set your OpenAI API key first:

bash

Collapse

Wrap

Copy
export OPENAI_API_KEY="your-key-here"
Run it, and it’ll echo “Hello!” with some AI flair (e.g., “Hey there, Hello! What a blast!”).

Directory Structure
text

Collapse

Wrap

Copy
robai/
├── __init__.py     # Exports core classes and utilities
├── base.py         # BaseRobot, BaseAI, and PromptManager
├── chat.py         # ChatRobot for conversational bots
├── errors.py       # Custom exceptions
├── func_tools.py   # Decorators for function control
├── protocols.py    # MessageHandler for I/O
├── schemas.py      # Pydantic models for messages and events
├── tools.py        # ToolSet for function management
├── utility.py      # Logging and helper functions
└── setup.py        # Package setup
Key Components
BaseRobot
The heart of robai. Subclass it to define your bot’s behavior:

prepare(): Set up input and prompts.
process(): Handle AI responses (streaming or non-streaming).
stop_condition(): When to halt (customize this!).
Functions: Mark methods with @robot_function for AI to call.
PromptManager
Manages chat history:

Keeps system messages (e.g., “You’re a helpful bot”).
Trims old messages based on max_messages (default 50) or max_tokens (default 4000).
MessageHandler
Handles I/O:

ConsoleMessageHandler: Prints to terminal (great for testing).
WebSocketMessageHandler: Streams to WebSocket clients.
Extend it for custom needs (e.g., GUI integration).
Example with GUI (BeeWare)
Integrate with Toga for a test window:

python

Collapse

Wrap

Copy
import toga
from toga.style import Pack, COLUMN
from robai.base import BaseRobot
from robai.schemas import ChatMessage, SystemMessage

class AIRobot(BaseRobot):
    def _init_system_prompt(self):
        self.system_prompt = SystemMessage(content="You’re a friendly AI!")

    async def prepare(self):
        self.prompt_manager.add_message(self.input_data)

    async def process(self):
        await self.generate_ai_response()
        await self._handle_streaming_response()

    async def stop_condition(self):
        return True  # One-shot response

def build(app):
    box = toga.Box(style=Pack(direction=COLUMN, padding=10))
    input_field = toga.TextInput(placeholder="Ask me anything!", style=Pack(padding=5))
    response_label = toga.Label("Response: (waiting)", style=Pack(padding=5))

    async def process_input(widget):
        text = input_field.value
        if text:
            bot = AIRobot()
            await bot.load(ChatMessage(content=text))
            await bot.interact()
            response_label.text = f"Response: {bot.output_data.content}"
            input_field.value = ""

    button = toga.Button("Submit", on_press=process_input, style=Pack(padding=5))
    box.add(input_field)
    box.add(button)
    box.add(response_label)
    return box

def main():
    return toga.App("AI App", "org.example.ai", startup=build)

if __name__ == "__main__":
    main().main_loop()
Run with briefcase dev—type, click, and see the AI respond in the window!

Usage Tips
Streaming: Set stream=True in BaseRobot for real-time responses.
Functions: Use @robot_function to let the AI call your methods:
python

Collapse

Wrap

Copy
@robot_function()
async def say_hello(self, name: str):
    return f"Hello, {name}!"
Testing: Call interact(test=True) for an interactive console mode.
Requirements
An OpenAI API key (set via OPENAI_API_KEY).
Python 3.11+.
Dependencies listed in setup.py.
Contributing
Fork github.com/philmade/robai, tweak, and PR! Issues and feature requests are welcome.

Why RobAI?
It’s lightweight, flexible, and feels like Python—none of that “dependency hell” nonsense. Pair it with a GUI like Toga, and you’ve got a clean, AI-driven app in minutes. Perfect for your test window dreams!