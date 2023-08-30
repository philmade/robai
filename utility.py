from pprint import pformat
from typing import Any

from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import PythonLexer
from pygments.styles import get_style_by_name
from faker import Faker
from robai.in_out import ChatMessage
from rich.panel import Panel
from rich.console import Console
from rich.theme import Theme

fake = Faker()


class FakeCompletion:
    def create(*args, **kwargs):
        fake_response = {
            "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
            "object": "text_completion",
            "created": 1589478378,
            "model": "text-davinci-003",
            "choices": [
                {
                    "text": " ".join(fake.sentences(nb=3)),
                    "index": 0,
                    "logprobs": "null",
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        }
        return fake_response


class FakeChatCompletion:
    def create(*args, **kwargs):
        fake_response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": " ".join(fake.sentences(nb=7)),
                        "role": "assistant",
                    },
                }
            ],
            "created": 1677664795,
            "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
            "model": "gpt-3.5-turbo-0613",
            "object": "chat.completion",
            "usage": {"completion_tokens": 17, "prompt_tokens": 57, "total_tokens": 74},
        }
        return fake_response


class InteractiveFakeCompletion:
    def create(*args, **kwargs):
        print("Request to OpenAI's Text Completion:")
        print(kwargs)  # Displaying the request object for clarity
        user_response = input("Enter the AI's completion: ")

        fake_response = {
            "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
            "object": "text_completion",
            "created": 1589478378,
            "model": "text-davinci-003",
            "choices": [
                {
                    "text": user_response,
                    "index": 0,
                    "logprobs": "null",
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": len(user_response.split()),
                "total_tokens": len(user_response.split()) + 5,
            },
        }
        return fake_response


class InteractiveFakeChatCompletion:
    def create(*args, **kwargs):
        print("Request to OpenAI's Chat Completion:")
        print(kwargs)  # Displaying the request object for clarity
        user_response = input("Enter the AI's chat completion: ")

        fake_response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": user_response,
                        "role": "assistant",
                    },
                }
            ],
            "created": 1677664795,
            "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
            "model": "gpt-3.5-turbo-0613",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": len(user_response.split()),
                "prompt_tokens": 57,
                "total_tokens": len(user_response.split()) + 57,
            },
        }
        return fake_response


class interactiveOpenAI:
    api_key = "dummy"
    completion = InteractiveFakeCompletion()
    ChatCompletion = InteractiveFakeChatCompletion()


class fakeOpenAI:
    api_key = "dummy"
    Completion = FakeCompletion()
    ChatCompletion = FakeChatCompletion()


def pprint_color(obj: Any, style_name: str = "github-dark") -> None:
    """Pretty-print in color."""
    from pygments import styles

    styles.STYLE_MAP
    style = get_style_by_name(style_name)
    formatter = Terminal256Formatter(style=style)
    formatted_string = pformat(obj)
    formatted_string = formatted_string.strip("'\n")
    formatted_string = formatted_string.replace("\\n", "\n")
    print(highlight(formatted_string, PythonLexer(), formatter), end="")


class MessagePrinter:
    def __init__(self):
        self.color_state = False
        theme = Theme(
            {
                "red": "red",
                "blue": "blue",
                "green": "green",
                "yellow": "yellow",
                "cyan": "cyan",
                "magenta": "magenta",
            }
        )
        self.console = Console(theme=theme, width=100)

    def toggle_color(self):
        """Toggles between two states, allowing us to alternate colors."""
        self.color_state = not self.color_state

    def get_color(self):
        return "\033[32m" if self.color_state else "\033[34m"

    def get_color(self, robot: "AIRobot") -> str:
        colors = {
            "red": "\033[91m",
            "blue": "\033[94m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "cyan": "\033[96m",
            "magenta": "\033[95m",
        }
        return colors.get(robot.color, "\033[0m")  # default to reset color

    def reset_color(self):
        return "\033[0m"

    def pprint_color(self, obj: Any, style_name: str = "github-dark") -> None:
        """Pretty-print in color."""
        from pygments import styles

        styles.STYLE_MAP
        style = get_style_by_name(style_name)
        formatter = Terminal256Formatter(style=style)
        formatted_string = pformat(obj)
        formatted_string = formatted_string.strip("'\n")
        formatted_string = formatted_string.replace("\\n", "\n")
        print(highlight(formatted_string, PythonLexer(), formatter), end="")


# Simple fake DB
class FakeDB:
    def similar_clothes(query: str):
        return ["dark denim jeans", "Aztec print jacket", "white t-shirt"]

    def store_the_db(thing: dict):
        return "Success"
