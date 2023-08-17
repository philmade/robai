from .languagemodels import BaseAIModel
from .errors import AIRobotInitializationError, SerializationError
from .memory import BaseMemory
from .chains import do_nothing_in_post_call
from typing import Any, List, Callable, Generator
from pydantic import BaseModel
import inspect
from loguru import logger
import asyncio
import json


class AIRobot:
    def __init__(
        self,
        memory: BaseMemory,
        ai_model: BaseAIModel,
        pre_call_chain: List[Callable],
        post_call_chain: List[Callable] = do_nothing_in_post_call,
        robots_functions: List[Callable] = None,
        **kwargs,
    ):
        self.pre_call_chain = pre_call_chain
        self.post_call_chain = post_call_chain
        self.robots_functions = robots_functions or []
        self.memory: BaseMemory = memory
        if memory.purpose is None or "":
            raise AIRobotInitializationError(
                message="Memory must have a purpose to be used by a robot"
            )
        self.ai_model: BaseAIModel = ai_model()
        # Check if the type annotations match
        # Check if the type annotations match
        memory_instruction_type = self.memory.__annotations__.get(
            "instructions_for_ai", None
        )
        ai_model_instruction_type = self.ai_model.instructions_for_ai_type

        if memory_instruction_type != ai_model_instruction_type:
            raise TypeError(
                f"Your {self.memory.__class__.__name__}'s attribute 'instructions_for_ai' is of type ({memory_instruction_type}) \n"
                f"But the robot's ai_model {self.ai_model.__class__.__name__} is expecting ({ai_model_instruction_type}). Its defined at ai_model.instructions_for_ai_type\n"
                f"You likely need to change your memory module to be annotated to return a type matching what the ai_model expects."
            )

    def process(self, input_data: Any, stream: bool = False) -> BaseModel:
        expected_type = self.memory.__annotations__["input_model"]
        if not isinstance(input_data, expected_type):
            raise TypeError(
                f"Expected input of type {expected_type}, but got {type(input_data)}"
            )
        # IMPORTANT STEP - THE INPUT MUST BE ADDED TO THE MEMORY
        self.memory.input_model = input_data
        # THE USER CAN DO WHATEVER THEY WANT FROM IT FROM THERE
        self.memory.complete = False
        while not self.memory.complete:
            # Pre-call chain
            for function in self.pre_call_chain:
                try:
                    if inspect.isgeneratorfunction(function):  # Generator function
                        self.memory.set_complete()
                        gen = function(self.memory)
                        input_model_processed = [
                            item for item in gen
                        ]  # Consume generator

                    else:  # Regular synchronous function
                        self.memory = function(self.memory)
                except Exception as e:
                    logger.info(
                        f"Exception in {function.__name__} in pre-call chain: {e}"
                    )

            try:
                json.dumps(self.memory.instructions_for_ai)
            except TypeError:
                raise SerializationError(
                    message=f"Your final function in pre-call isn't setting memory.instructions_for_ai to something that's JSON serializable, got: {self.memory.instructions_for_ai} instead"
                )

            # Call to AI model
            if stream:
                streamed_response: Generator = self.ai_model.get_ai_response(
                    self.memory
                )
            else:
                self.memory = self.ai_model.get_ai_response(self.memory)

            # Post-call chain
            for function in self.post_call_chain:
                try:
                    if inspect.isgeneratorfunction(function):  # Generator function
                        self.memory.set_complete()
                        gen = function(streamed_response, self.memory)
                        return gen
                    else:  # Regular synchronous function
                        self.memory = function(self.memory)
                except Exception as e:
                    logger.info(
                        f"Exception in {function.__name__} in post-call chain: {e}"
                    )

            return self.memory

    async def aprocess(self, input_data: Any, stream: bool = False) -> BaseModel:
        if not isinstance(input_data, type(self.memory.input_model)):
            raise TypeError(
                f"Robot recieved {type(input_data)} as input, but it expected what's in memory.input_model which is type {type(self.memory.input_model)}"
            )
        # IMPORTANT STEP - THE INPUT MUST BE ADDED TO THE MEMORY
        self.memory.input_model = input_data
        # THE USER CAN DO WHATEVER THEY WANT FROM IT FROM THERE
        self.memory.complete = False
        while not self.memory.complete:
            # Pre-call chain
            for function in self.pre_call_chain:
                try:
                    if asyncio.iscoroutinefunction(function):  # Async function
                        self.memory = await function(self.memory)
                    elif inspect.isgeneratorfunction(function):  # Generator function
                        self.memory.set_complete()
                        gen = function(self.memory)
                        input_model_processed = [
                            item for item in gen
                        ]  # Consume generator

                    else:  # Regular synchronous function
                        self.memory = function(self.memory)
                except Exception as e:
                    logger.info(
                        f"Exception in {function.__name__} in pre-call chain: {e}"
                    )

            try:
                json.dumps(self.memory.instructions_for_ai)
            except TypeError:
                raise SerializationError(
                    message=f"Your final function in pre-call isn't setting memory.instructions_for_ai to something that's JSON serializable, got: {input_model_processed} instead"
                )

            # Call to AI model
            if stream:
                streamed_response: Generator = self.ai_model.stream_call(self.memory)
            else:
                self.memory: BaseMemory = self.ai_model.call(self.memory)

            # Post-call chain
            for function in self.post_call_chain:
                try:
                    if asyncio.iscoroutinefunction(function):  # Async function
                        self.memory = await function(self.memory)
                    elif inspect.isgeneratorfunction(function):  # Generator function
                        self.memory.set_complete()
                        gen = function(streamed_response, self.memory)
                        return gen
                    else:  # Regular synchronous function
                        self.memory = function(self.memory)
                except Exception as e:
                    logger.info(
                        f"Exception in {function.__name__} in post-call chain: {e}"
                    )

        return self.memory
