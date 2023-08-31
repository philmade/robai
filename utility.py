from pprint import pformat
from typing import Any

from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import PythonLexer
from pygments.styles import get_style_by_name
from faker import Faker
from robai.in_out import ChatMessage
from rich.console import Console
from rich.pretty import pprint

fake = Faker()


class CustomConsole(Console):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pprint(self, message: object) -> None:
        """
        Prints the message in a visually appealing format with alternating colors.
        Also, provides a clean copyable version of the content.
        """
        pprint(console=self, _object=message)

    def pprint_message(self, message: ChatMessage, robot: "AIRobot" = None) -> None:
        """
        Prints the message in a visually appealing format with alternating colors.
        Also, provides a clean copyable version of the content.
        """
        if not robot:
            robot = self

        robot_name = robot.__class__.__name__
        self.rule(f"{robot_name}")
        pprint(console=self, _object=message)


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


# Simple fake DB
class FakeDB:
    def similar_clothes(query: str):
        return ["dark denim jeans", "Aztec print jacket", "white t-shirt"]

    def store_the_db(thing: dict):
        return "Success"
