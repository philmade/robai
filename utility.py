from faker import Faker
from typing import Callable, List
from inspect import signature, Parameter
from typing import get_type_hints
from functools import wraps

fake = Faker()


class AIDescription:
    def __init__(self, type: str, description: str, enum: List[str] = None):
        self.type: str = type
        self.description: str = description
        self.enum: List[str] = enum

    def to_dict(self):
        result = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            result["enum"] = self.enum
        return result


def add_description(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(
            f"Wrapper called with args: {args}, kwargs: {kwargs}"
        )  # Debugging statement
        return func(*args, **kwargs)

    def description_func():
        sig = signature(func)
        params = sig.parameters
        type_hints = get_type_hints(func)

        properties = {
            name: type_hints[name].to_dict()
            for name, param in params.items()
            if name != "self" and isinstance(type_hints.get(name), AIDescription)
        }

        required = [
            name
            for name, param in params.items()
            if param.default == Parameter.empty
            and name != "self"
            and isinstance(type_hints.get(name), AIDescription)
        ]

        description = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        return description

    wrapper.description = description_func
    return wrapper


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]
