from robai.schemas import MarkdownFunctionResults
from typing import List, Callable, Dict, Optional, Union
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def set_next_functions(*next_function_names: str) -> Callable:
    """Decorator that sets which functions will be available after this function is called.
    Takes function names as strings, which will be looked up from self.functions.
    This is mutually exclusive with @reset_after_use and @remove_after_use."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            try:
                result = await func(self, *args, **kwargs)
                # After function completes successfully, look up and set the next available functions
                next_functions = []
                logger.debug(f"Setting next functions: {next_function_names}")
                logger.debug(f"Available functions: {list(self.functions.keys())}")
                for name in next_function_names:
                    if name in self.functions:
                        next_functions.append(self.functions[name])
                        logger.debug(
                            f"Added function {name} to next available functions"
                        )
                    else:
                        logger.error(f"Function {name} not found in toolset")
                        raise ValueError(f"Function {name} not found in toolset")
                self.available_functions = next_functions
                logger.debug(
                    f"Set available functions to: {[f.__name__ for f in next_functions]}"
                )
                return result
            except Exception as e:
                # On error, reset to default state
                self.reset_available_functions()
                raise e

        return wrapper

    return decorator


def remove_after_use(func: Callable) -> Callable:
    """Decorator to remove a function from available_functions after it's used.
    First resets all functions to default state, then removes itself.
    This is mutually exclusive with @set_next_functions and @reset_after_use."""

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            result = await func(self, *args, **kwargs)
            # First reset to default state
            self.reset_available_functions()
            # Then remove this function from the list
            self.available_functions = [
                f for f in self.available_functions if f.__name__ != func.__name__
            ]
            return result
        except Exception as e:
            # On error, make sure we reset to a clean state
            self.reset_available_functions()
            raise e

    return wrapper


def reset_after_use(func: Callable) -> Callable:
    """Decorator to reset all available functions after use.
    This is mutually exclusive with @set_next_functions and @remove_after_use."""

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            result = await func(self, *args, **kwargs)
            # Reset all available functions after successful execution
            self.reset_available_functions()
            return result
        except Exception as e:
            # On error, make sure we reset to a clean state
            self.reset_available_functions()
            raise e

    return wrapper


class ToolSet:
    def __init__(self):
        self.functions = {}  # Keep this as a dict for lookup
        self.available_functions = []  # Change to list for simpler management

    def set_functions(self, functions: List[Callable]) -> None:
        """Set the initial functions and available functions"""
        self.functions = {func.__name__: func for func in functions}
        self.available_functions = functions.copy()

    def get_available_functions(self) -> List[Callable]:
        """Get currently available functions for the AI"""
        return self.available_functions

    def reset_available_functions(self) -> None:
        """Reset available functions to all functions"""
        self.available_functions = list(self.functions.values())

    def remove_current_function(self, function_name: str) -> None:
        """Remove a function from available functions"""
        self.available_functions = [
            f for f in self.available_functions if f.__name__ != function_name
        ]
