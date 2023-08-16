from .llm import OpenAIChatCompletion, BaseAIModel
from .errors import AIRobotInitializationError
from .memory import BaseMemory
from .chains import do_nothing
from typing import Any, List, Callable, Tuple
from pydantic import BaseModel
import inspect
from loguru import logger
import asyncio


class AIRobot:
    def __init__(
        self,
        memory: BaseMemory,
        input_model: Any,
        output_model: Any,
        pre_call_chain: List[Callable],
        post_call_chain: List[Callable] = do_nothing,
        robots_functions: List[Callable] = None,
        ai_model: BaseAIModel = OpenAIChatCompletion,
        **kwargs,
    ):
        self.input_model = input_model
        self.output_model = output_model
        self.pre_call_chain = pre_call_chain
        self.post_call_chain = post_call_chain
        self.robots_functions = robots_functions or []
        self.memory = memory
        if memory.purpose is None or "":
            raise AIRobotInitializationError(
                message="Memory must have a purpose to be used by a robot"
            )
        self.ai_model: OpenAIChatCompletion = ai_model()
        self.ai_model.setup(max_tokens=100)
        super().__init__(**kwargs)
        self._validate_last_function_in_pre_call_chain()

    def _validate_last_function_in_pre_call_chain(self):
        # Base serializable types
        serializable_types = (str, dict, list, int, float, bool, type(None))

        # Recognizable typing types
        serializable_typing_origins = (list, dict)

        # Get the return annotation of the last function in the pre_call_chain
        return_annotation = self.pre_call_chain[-1].__annotations__.get("return")
        return_annotation = return_annotation.__args__[0]

        # Direct type check
        if (
            isinstance(return_annotation, type)
            and return_annotation in serializable_types
        ):
            return

        # Check for generic typing aliases like List[dict]
        if (
            hasattr(return_annotation, "__origin__")
            and return_annotation.__origin__ in serializable_typing_origins
        ):
            return

        # If none of the above checks pass
        raise AIRobotInitializationError(
            message="The final function in the pre_call_chain must return a JSON serializable type."
        )

    def pre_call(self, input_data: BaseModel) -> Tuple[BaseModel, BaseModel]:
        input_model_processed = self.input_model(**input_data.dict())
        for function in self.pre_call_chain:
            input_model_processed, self.memory = function(
                input_model_processed, self.memory
            )
        return input_model_processed, self.memory

    def post_call(self, ai_response: BaseModel) -> Tuple[Any, BaseModel]:
        if self.post_call_chain:
            ai_response_processed = self.output_model(**ai_response.dict())
            for function in self.post_call_chain:
                ai_response_processed, self.memory = function(
                    ai_response_processed, self.memory
                )
            return ai_response_processed, self.memory
        else:
            return ai_response, self.memory

    def process(
        self, input_data: BaseModel, stream: bool = True
    ) -> BaseModel:  # self.output_model
        while not self.memory.complete:
            input_model_processed, self.memory = self.pre_call(input_data)
            if stream:
                ai_response_processed = self.ai_model.stream_call(input_model_processed)
            else:
                ai_response_processed = self.ai_model.call(input_model_processed)
            ai_response_processed, self.memory = self.post_call(ai_response_processed)

        return ai_response_processed

    async def aprocess(self, input_data: BaseModel, stream: bool = True) -> BaseModel:
        self.memory.complete = False
        while not self.memory.complete:
            # Pre-call chain
            for function in self.pre_call_chain:
                try:
                    if asyncio.iscoroutinefunction(function):  # Async function
                        input_model_processed, self.memory = await function(
                            input_data, self.memory
                        )
                    elif inspect.isgeneratorfunction(function):  # Generator function
                        self.memory.set_complete()
                        gen = function(input_model_processed, self.memory)
                        input_model_processed = [
                            item async for item in gen
                        ]  # Consume generator

                    else:  # Regular synchronous function
                        input_model_processed, self.memory = function(
                            input_model_processed, self.memory
                        )
                except Exception as e:
                    logger.info(
                        f"Exception in {function.__name__} in pre-call chain: {e}"
                    )

            # Call to AI model
            if stream:
                ai_response_processed = self.ai_model.stream_call(input_model_processed)
            else:
                ai_response_processed = self.ai_model.call(input_model_processed)

            # Post-call chain
            for function in self.post_call_chain:
                try:
                    if asyncio.iscoroutinefunction(function):  # Async function
                        ai_response_processed, self.memory = await function(
                            ai_response_processed, self.memory
                        )
                    elif inspect.isgeneratorfunction(function):  # Generator function
                        self.memory.set_complete()
                        gen = function(ai_response_processed, self.memory)
                        ai_response_processed = gen  # Consume generator
                    else:  # Regular synchronous function
                        ai_response_processed, self.memory = function(
                            ai_response_processed, self.memory
                        )
                except Exception as e:
                    logger.info(
                        f"Exception in {function.__name__} in post-call chain: {e}"
                    )

        return ai_response_processed
