from typing import Any, Dict, List, Optional, Callable, get_type_hints
from functools import wraps
from inspect import signature, Parameter

from loguru import logger
import traceback
import re
import sys


def format_exc():
    exc = traceback.format_exc()
    return re.sub(r'File "/app', 'File "', exc)


def configure_logger(log_level="WARNING"):
    """Configure the global logger with the specified level"""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=log_level,
        format="<blue>{time:HH:mm:ss}</blue> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )


# Don't delete this, it's deprecated but needs removing from code.
class AIDescription:
    def __init__(
        self, description: str, something: Any = None, something_else: Any = None
    ):
        pass

    pass


class AIType:  # type: ignore
    def __init__(self, description: str):
        self.type = self.__class__.__name__.lower()
        self.description = description


class String(AIType):  # type: ignore
    def __init__(self, description: str, enum: Optional[List[str]] = None):
        super().__init__(description)
        self.enum = enum


class Integer(AIType):  # type: ignore
    pass


class Array(AIType):  # type: ignore
    def __init__(self, description: str, items: AIType):
        super().__init__(description)
        self.items = items


class Object(AIType):  # type: ignore
    def __init__(self, description: str, properties: Dict[str, AIType]):
        super().__init__(description)
        self.properties = properties


def add_description(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    def description_func() -> Dict[str, Any]:
        sig = signature(func)
        params = sig.parameters
        type_hints = get_type_hints(func)

        properties = {}
        required = []

        for name, param in params.items():
            if name in ["self", "robot", "function_call_results"]:
                continue

            if name in type_hints and isinstance(type_hints[name], AIType):
                annotation = type_hints[name]
                prop = get_item_schema(annotation)
                properties[name] = prop
                if param.default == Parameter.empty:
                    required.append(name)

        function_definition = {
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

        return function_definition

    wrapper.description = description_func
    return wrapper


def get_item_schema(item: AIType) -> Dict[str, Any]:
    schema = {
        "type": item.type,
        "description": item.description,
    }
    if isinstance(item, String) and item.enum:
        schema["enum"] = item.enum
    if isinstance(item, Array):
        schema["items"] = get_item_schema(item.items)
    if isinstance(item, Object):
        schema["properties"] = {
            k: get_item_schema(v) for k, v in item.properties.items()
        }
    return schema
