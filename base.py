from typing import (
    AsyncGenerator,
    List,
    Self,
    Any,
    TypeVar,
    Generic,
    Union,
    Callable,
    Iterable,
    Tuple,
    Dict,
)
from openai import AsyncOpenAI
import openai
from openai.types.chat.chat_completion import ChatCompletion
import os
from abc import ABC, abstractmethod
from loguru import logger
import traceback
import re
from robai.chat import HistoryManager, LocalHistoryManager
from robai.schemas import ChatMessage, AIMessage, SystemMessage, FunctionResults


T = TypeVar("T")


def format_exc():
    exc = traceback.format_exc()
    return re.sub(r'File "/app', 'File "', exc)


class BaseAI:
    def __init__(self):
        API_KEY = os.getenv("OPENAI_API_KEY")
        if API_KEY is None:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.openai = openai
        self.openai.api_key = API_KEY
        self.aclient: AsyncOpenAI = AsyncOpenAI(api_key=API_KEY)
        self.models: List = self.aclient.models.list()
        self.model: str = "gpt-3.5-turbo"
        self.output_date: Union[str, AsyncGenerator[str, None]] = None
        self.stream: bool = False
        self.max_tokens: int = 100

    async def generate_response(
        self,
        prompt: List[ChatMessage],
        tools_for_ai: Dict[str, Callable] = None,
        *args,
        **kwargs,
    ) -> Tuple[Union[AsyncGenerator[str, None], str], Union[Iterable, None]]:
        messages = [message.model_dump() for message in prompt]

        # Ensure only 'role' and 'content' keys are in the dictionaries
        messages = [
            {k: v for k, v in message.items() if k in ["role", "content"]}
            for message in messages
        ]
        if tools_for_ai:
            tools = [tool.description() for tool in tools_for_ai.values()]
        else:
            tools = None
        if self.stream:
            response_generator = await self.aclient.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                n=1,
                stop=None,
                temperature=0.7,
                stream=self.stream,
                tools=tools,
            )
            return response_generator
        else:
            response: ChatCompletion = await self.aclient.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                n=1,
                stop=None,
                temperature=0.7,
                stream=self.stream,
                tools=tools,
            )
            logger.info("Finish reason: " + response.choices[0].finish_reason)
            return (
                response.choices[0].message.content,
                response.choices[0].message.tool_calls,
            )


class BaseRobot(ABC, Generic[T]):
    def __init__(
        self,
        ai_class: BaseAI = BaseAI(),
        ai_model: str = "gpt-4o",
        stream: bool = True,
        max_tokens: int = 100,
    ) -> Self:
        self.input_data: Any = None
        self.stream: bool = stream
        self.ai_class = ai_class
        self.ai_class.model = ai_model
        self.ai_class.stream = stream
        self.ai_class.max_tokens = max_tokens
        self.prompt: List[ChatMessage] = []
        self.system_prompt: SystemMessage = None
        self.input_data: T = None
        self.output_data: T = None
        self.robot_name: str = self.__class__.__name__
        self.history = LocalHistoryManager(robot_name=self.robot_name)
        self.tools: Dict[str, Callable] = {}
        self.tool_call_results: Dict[str, str] = {}

    # @abstractmethod
    def load(self, input_data: Any, *args, **kwargs):
        """
        This has to be implemented by the child class.
        It should at least take input_data argument of type Any, and set it to self.input_data
        For any other data your robot needs to work, load it here, and set them as class attributes.
        Finally, you must use this data to set the system_prompt attribute as type BaseRobot.SystemMessage
        Example:
        def load(self, input_data: ChatMessage, db_user: DBUser, db_session: AsyncSession):
            self.input_data = input_data
            self.db_user = db_user
            self.db_meta = db_meta
            self.system_prompt = BaseRobot.SystemMessage(role="system", content=f"{self.db_user.username} will be talking with you. His interests are {self.db_meta}")
            logger.info(f"Loaded {self.input_data},{self.db_user},{self.db_meta}")
            logger.info(f"System prompt: {self.system_prompt}")
        Now in your before_call() and after_call() methods, you can access self.db_user and self.db_session
        """
        pass

    @abstractmethod
    def stop_condition(self) -> bool:
        """
        This has to be implemented by the child class.
        It should return a boolean value indicating whether the interaction should stop.
        Example:
        def stop_condition(self) -> bool:
            return len(self.conversation_history) >= 5
        """
        pass

    async def before_call(self, input_data: Any) -> None:
        pass

    async def after_call(self, output_data: Any) -> None:
        pass

    async def log_input(self, input_data):
        print(f"\033[94m[{self}] Input:\033[0m {input_data}")

    async def log_thinking(self):
        print(f"\033[93m[{self}] Thinking...\033[0m")

    async def log_output(self, output_data):
        print(f"\033[92m[{self}] Output:\033[0m {output_data}")

    async def log_message(self, message):
        print(f"\033[96m[{self}] Message:\033[0m {message}")

    async def interact(
        self,
        # ) -> Union[str, AsyncGenerator[str, None]]:
    ) -> T:
        try:
            await self.before_call()

            if self.prompt is []:
                raise ValueError(
                    "Prompt not set - you must set a prompt in before_call() or the AI does nothing"
                )

            if self.tools:
                for _func in self.tools.values():
                    if not hasattr(_func, "description"):
                        raise ValueError(
                            "Tools must have a description method that returns a string. Decorate func with robai.utility.add_description"
                        )
            tools = self.tools if self.tools else None
            tool_call_results = (
                self.tool_call_results if self.tool_call_results else None
            )

            await self.log_message(self.prompt)

            if self.ai_class.stream:
                # We set output to a generator, which users can handle in post-call

                response_generator = await self.ai_class.generate_response(
                    self.prompt, tools, tool_call_results
                )
                self.output_data = response_generator
            else:
                # Else we wait for output as a string from the model
                output_data, tool_calls = await self.ai_class.generate_response(
                    self.prompt, tools, tool_call_results
                )
                ai_response = AIMessage(role="assistant", content=output_data)
                self.output_data = ai_response

            await self.log_output(f"Output: {self.output_data}")
            await self.after_call()

            if await self.stop_condition() is False:
                return await self.interact()

            return self.output_data

        except (NameError, AttributeError) as e:
            logger.error(f"{e}\n{format_exc()}")
            logger.info(
                "Did you call load() before running the robot? Did you set the system_prompt?"
            )
        except Exception as e:
            logger.error(f"Exception occurred: {e}\n{format_exc()}")
