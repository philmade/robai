from abc import ABC, abstractmethod
from typing import Generator
from faker import Faker
import time
from .robot_in import ChatMessage
from .memory import BaseMemory

fake = Faker()

# TYPE MAPPINGS FOR CLARITY OF RETURNS
string_response = str


class BaseAIModel(ABC):
    """
    Base class for AI models.
    1) On init, you must pass your robots input model
    2) Init the calls .prep_for_ai() on your input model
    3) Call .run() on your robot
    4) The robot will call .setup() and .call() on your robot
    5) Call() usees self.prepped_input as the prompt
    6) Call() returns a string - parsing is done at the robot level
    """

    is_chat_model: bool = False

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def setup(self, **kwargs) -> None:
        """
        Setup the language model with the provided parameters.
        MUST BE CALLED BEFORE USING THE MODEL
        """
        pass

    @abstractmethod
    def call(self, memory: BaseMemory, **kwargs) -> BaseMemory:
        """
        Call the language model with the provided prompt and return the generated text.
        """
        pass

    def stream_call(self, memory: BaseMemory, **kwargs) -> Generator:
        """
        Call the language model with the provided prompt and return the generated text.
        """
        pass


class FakeAICompletion(BaseAIModel):
    is_ready: bool = False

    def setup(self, *args, **kwargs):
        self.is_ready = True

    def call(self, memory: BaseMemory):
        if not self.is_ready:
            raise Exception("Model is not ready - YOU MUST CALL SETUP")
        message = ChatMessage(role="assistant", content=fake.sentences(nb=70))
        memory.add_message(message)
        return memory

    def stream_call(self, memory: BaseMemory) -> Generator:
        if not self.is_ready:
            raise Exception("Model is not ready - YOU MUST CALL SETUP")
        for sentence in fake.sentences(nb=70):
            streamed_response = sentence
            time.sleep(0.05)
            yield streamed_response
