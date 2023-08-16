from .llm import BaseAIModel
from .errors import AIRobotInitializationError, SerializationError
from .memory import BaseMemory
from .chains import do_nothing
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
        post_call_chain: List[Callable] = do_nothing,
        robots_functions: List[Callable] = None,
        **kwargs,
    ):
        self.pre_call_chain = pre_call_chain
        self.post_call_chain = post_call_chain
        self.robots_functions = robots_functions or []
        self.memory = memory
        if memory.purpose is None or "":
            raise AIRobotInitializationError(
                message="Memory must have a purpose to be used by a robot"
            )
        self.ai_model: BaseAIModel = ai_model()
        self.ai_model.setup(max_tokens=100)

    def process(self, input_data: Any, stream: bool = True) -> BaseModel:
        if not isinstance(input_data, type(self.memory.input_model)):
            raise AIRobotInitializationError(
                message=f"Input data must be of type {self.memory.input_model} Make sure your memory class has an input_model, and make sure an instance of this instance_model is sent to aprocess"
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

    async def aprocess(self, input_data: Any, stream: bool = True) -> BaseModel:
        if not isinstance(input_data, type(self.memory.input_model)):
            raise AIRobotInitializationError(
                message=f"Input data must be of type {self.memory.input_model} Make sure your memory class has an input_model, and make sure an instance of this instance_model is sent to aprocess"
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
