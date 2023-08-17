from abc import ABC, abstractmethod
from inspect import signature
from faker import Faker
import time
import openai
import os
from typing import Any, Union, Optional, List, Generator
from .in_out import ChatMessage
from .memory import BaseMemory

fake = Faker()


class BaseAIModel(ABC):
    instructions_for_ai_type: Any
    # any_properties_for_the_ai_model, like max_tokens, temperature, etc
    # Set them here, and then use self.some_property in the call method

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def call(self, memory: BaseMemory) -> BaseMemory:
        # call the AI model and get a response
        memory.ai_string_response = "This is the string version of the AI response"
        memory.ai_raw_response = "This is the response object, if available"
        return memory

    def stream_call(self, memory: BaseMemory) -> Generator:
        pass

    def get_ai_response(
        self, memory: BaseMemory, stream=False, **kwargs
    ) -> BaseMemory | Generator:
        #         if not isinstance(
        #             memory.instructions_for_ai, type(self.instructions_for_ai_type)
        #         ):
        #             raise TypeError(
        #                 f"""The AI model expects input of type {type(self.instructions_for_ai_type)},
        #                 but got your precall chain returned memory.instructions_for_ai of type {type(memory.instructions_for_ai)}
        # """
        #             )
        if stream:
            generator_response = self.stream_call(memory)
            return generator_response
        else:
            memory = self.call(memory)
            message = ChatMessage(role="assistant", content=memory.ai_string_response)
            memory.ai_response = message
            return memory

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Check for the presence of required class attributes
        if not hasattr(cls, "instructions_for_ai_type"):
            raise TypeError(
                f"""Your AI model {cls.__name__} is missing 'instructions_for_ai_type' attribute. 
                What does your ai_model expect to recieve? Eg OpenAIChatCompletion expects an array of message objects
                """
            )

        # Check if abstract methods in subclasses accept the same arguments as the base class
        base_methods = [BaseAIModel.call]
        for base_method in base_methods:
            base_signature = signature(base_method)
            subclass_method = getattr(cls, base_method.__name__)
            subclass_signature = signature(subclass_method)

            if base_signature.parameters != subclass_signature.parameters:
                raise TypeError(
                    f"Method {base_method.__name__} in subclass {cls.__name__} has an incorrect argument type. Look at the BaseAIModel.call method "
                    f"Expected {base_signature}, but got {subclass_signature}."
                )


class FakeAICompletion(BaseAIModel):
    instructions_for_ai_type = List[ChatMessage]

    def call(self, memory: BaseMemory) -> BaseMemory:
        # Here fake sentences would be the string response from the AI.
        fake_response = fake.sentences(nb=5)
        fake_response_string = " ".join(fake_response)
        # In any AI model, these are the two things you need to add to memory
        # The string response is the most important, have a look at what happens with get_ai_response()
        memory.ai_raw_response = fake_response
        memory.ai_string_response = fake_response_string
        # And now we return the memory object!
        return memory

    def stream_call(self, memory: BaseMemory) -> Generator:
        fake_response = " ".join(fake.sentences(nb=5))
        for word in fake_response.split(" "):
            streamed_response = word
            time.sleep(0.05)
            yield streamed_response


class OpenAICompletion(BaseAIModel):
    instructions_for_ai_type = str
    model: str = "gpt-3.5-turbo-16k"
    suffix: str = ""
    max_tokens: int = 1000
    temperature: float = 0.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of: int = 1
    # logit_bias: Optional[Dict[str, float]] = None,
    user: Optional[str] = None
    openai = openai

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        API_KEY = os.getenv("OPENAI_API_KEY")
        if API_KEY is None:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.openai.api_key = API_KEY

    def call(
        self,
        memory: BaseMemory,
    ) -> BaseMemory:
        response = self.openai.Completion.create(
            model=self.model,
            prompt=memory.instructions_for_ai,
            suffix=self.suffix,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.n,
            stream=self.stream,
            logprobs=self.logprobs,
            echo=self.echo,
            stop=self.stop,
            # presence_penalty=self.presence_penalty,
            # frequency_penalty=self.frequency_penalty,
            # best_of=self.best_of,
            # logit_bias=self.logit_bias,
            # user=self.user,
        )
        memory.ai_string_response = response.choices[0].text.strip()
        memory.ai_raw_response = response
        return memory


class OpenAIChatCompletion(OpenAICompletion):
    instructions_for_ai_type = List[ChatMessage]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, memory: BaseMemory) -> BaseMemory:
        response = self.openai.ChatCompletion.create(
            model=self.model,
            messages=memory.instructions_for_ai,
            temperature=self.temperature,
            stop=self.stop,
            max_tokens=self.max_tokens,
        )
        memory.ai_string_response = response.choices[0]["message"]["content"]
        memory.ai_raw_response = response
        return memory

    def stream_call(self, memory: BaseMemory) -> Generator:
        response = self.openai.ChatCompletion.create(
            model=self.model,
            messages=memory.instructions_for_ai,
            temperature=self.temperature,
            stop=self.stop,
            max_tokens=1000,
            stream=True,  # Enable streaming
        )

        # Yield each chunk of the response as it becomes available
        for chunk in response:
            streamed_response: str = (
                chunk.choices[0].delta.content
                if "content" in chunk.choices[0].delta
                else None
            )
            yield streamed_response
