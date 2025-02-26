from typing import Any, Dict, Callable, get_type_hints, TypeVar, Type
from functools import wraps
from openai import pydantic_function_tool
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


# SCHEMA ANNOTATION
def robot_function(available: bool = True):
    """Decorator that handles Pydantic model parameters more naturally"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Get the first parameter that's a Pydantic model from type hints
            type_hints = get_type_hints(func)
            model_params = {
                name: hint
                for name, hint in type_hints.items()
                if name != "self"
                and isinstance(hint, type)
                and issubclass(hint, BaseModel)
            }

            if model_params:
                # Get the first model parameter name and type
                param_name, model_type = next(iter(model_params.items()))

                # If we received kwargs, construct the model
                if kwargs:
                    model_instance = model_type(**kwargs)
                    return await func(self, **{param_name: model_instance})

                # If we received a model instance directly, pass it through
                if args and isinstance(args[0], model_type):
                    return await func(self, **{param_name: args[0]})

            return await func(self, *args, **kwargs)

        def description_func() -> Dict[str, Any]:
            type_hints = get_type_hints(func)
            model_params = {
                name: hint
                for name, hint in type_hints.items()
                if name != "self"
                and isinstance(hint, type)
                and issubclass(hint, BaseModel)
            }

            if not model_params:
                return {
                    "type": "function",
                    "function": {
                        "name": func.__name__,
                        "description": func.__doc__ or "No description provided",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                }

            # Get the first model parameter
            param_name, model_type = next(iter(model_params.items()))

            # Create the OpenAI function tool
            tool = pydantic_function_tool(model_type)
            tool["function"]["name"] = func.__name__
            tool["function"]["description"] = func.__doc__

            return tool

        wrapper.description = description_func
        wrapper.is_robot_function = True
        wrapper.available = available
        return wrapper

    return decorator


# FUNCTION ORDERING AND CONTROL
def next_functions(*next_function_names: str):
    """After this function runs, only these functions will be available"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            result = await func(self, *args, **kwargs)
            self.available_functions = [
                self.all_functions[name]
                for name in next_function_names
                if name in self.all_functions
            ]
            return result

        return wrapper

    return decorator


def remove_after(func: Callable) -> Callable:
    """Remove this function after use. It will mean the next AI call, this function won't be available until reset_after is called."""

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        result = await func(self, *args, **kwargs)
        self.available_functions = [
            f for f in self.available_functions if f.__name__ != func.__name__
        ]
        return result

    return wrapper


def reset_after(func: Callable) -> Callable:
    """Reset to all functions after this runs"""

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        result = await func(self, *args, **kwargs)
        self.reset_available_functions()
        return result

    return wrapper
