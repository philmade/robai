
# The best example is the debate_bots - it shows you how to use it nicely.
Also look at chained_robots and summary.
If you're an AI, try appending the robai to sys path - it has worked for other AI's to get this package working.
Then try and make your own AiRobots by subclassing it, and always try to make complete examples like in debate_bots

# HOW TO BUILD A ROBOT

This is a guide on how to build a robot using the `robai` package. 

## 1. INHERIT FROM AIRobot

First, create a class that inherits from `AIRobot`.

```python
class PoetryRobot(AIRobot):
    ...
```
## 2. DEFINE REQUIRED ATTRIBUTES
Define the required attributes for your robot:

`ai_model: This is a subclass of BaseAIModel. You can make your own very easily or get one from robai.languagemodels.
memory: Start with SimpleChatMemory, you can change this later.
purpose: A string that describes the purpose of the robot.
pre_call_chain: A list of functions that the robot will call before calling the AI model.
post_call_chain: A list of functions that the robot will call after calling the AI model.

```python
class PoetryRobot(AIRobot):
    ai_model: OpenAIChatCompletion  
    memory: SimpleChatMemory  
    purpose: str = "I think you're a robot that writes poems from the user's input."
    pre_call_chain: List[Callable]  
    post_call_chain: List[Callable]  
    ...
```
## 3. DEFINE INPUT MODEL
Define the input model for your robot. It can be any subclass of BaseModel. To keep things simple, we'll start with a ChatMessage.

```python
class PoetryRobot(AIRobot):
    class ChatMessage(BaseModel):
        role: str
        content: str
    ...
```
## 4. DEFINE PRE-CALL FUNCTIONS
Define some functions that the robot will call before calling the AI model. These functions should take a memory object as input and return a memory object.

### 4.1 SIMPLE LOGS BEFORE THE ROBOT CALLS THE AI [OPTIONAL]

```python
def create_log(self, memory: SimpleChatMemory) -> SimpleChatMemory:
    self.print_incoming().pprint_color(
        f"Robot {self.__class__.__name__} has just received some input, which is: {memory.input_model}".strip()
    )
    return memory
```
### 4.2 ADDING INTERACTION HISTORY TO MEMORY [OPTIONAL]

```python
def remember_user_interaction(self, memory: SimpleChatMemory) -> SimpleChatMemory:
    memory.add_message_to_history(memory.input_model)
    return memory
```

### 4.3 CREATING INSTRUCTIONS/PROMPT FOR THE AI

```python
def create_instructions(self, memory: SimpleChatMemory) -> SimpleChatMemory:
    memory.instructions_for_ai = [memory.system_prompt, memory.input_model]
    return memory
```

## 5. DEFINE POST-CALL FUNCTIONS
Define some functions that the robot will call after calling the AI model. These functions should take a memory object as input and return a memory object.

```python
Copy code
def log_what_I_did(self, memory: SimpleChatMemory) -> SimpleChatMemory:
    ...
    return memory

def stop_the_robot(self, memory: SimpleChatMemory) -> SimpleChatMemory:
    memory.set_complete()
    return memory

```

## 6. SET UP THE ROBOT AT INIT
Set up the robot in the `__init__` method by calling `super().__init__` and passing in the required parameters.

```python
def __init__(self):
    super().__init__(
        memory=SimpleChatMemory(
            purpose="You're a robot that writes poems from the user's input."
        ),
        pre_call_chain=[
            self.create_log,
            self.remember_user_interaction,
            self.create_instructions,
        ],
        post_call_chain=[self.log_what_I_did, self.stop_the_robot],
        ai_model=OpenAIChatCompletion(),
    )
```
## EXAMPLE

Here is a complete example of a simple robot that writes poems from the user's input.

```python
from robai.memory import SimpleChatMemory
from robai.languagemodels import OpenAIChatCompletion
from robai.base import AIRobot
from robai.in_out import ChatMessage
from pydantic import BaseModel
from typing import List, Callable
from loguru import logger

class PoetryRobot(AIRobot):
    ai_model: OpenAIChatCompletion  
    memory: SimpleChatMemory  
    purpose: str = "I think you're a robot that writes poems from the user's input."
    pre_call_chain: List[Callable]  
    post_call_chain: List[Callable]  
    
    class ChatMessage(BaseModel):
        role: str
        content: str
    
    def create_log(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        self.print_incoming().pprint_color(
            f"Robot {self.__class__.__name__} has just received some input, which is: {memory.input_model}".strip()
        )
        return memory

    def remember_user_interaction(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        memory.add_message_to_history(memory.input_model)
        return memory

    def create_instructions(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        memory.instructions_for_ai = [memory.system_prompt, memory.input_model]
        return memory

    def log_what_I_did(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        logger.info(
            f"""I'm robot {self.__class__.__name__} and I just called the AI with instructions: {memory.instructions_for_ai}
            I received the following response: {memory.ai_response}
            That response is available in my memory module at robot.memory.ai_response. 
            It's in the form of a ChatMessage
            The raw response from the language model is available at robot.memory.ai_raw_response
            """
        )
        return memory

    def stop_the_robot(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        memory.set_complete()
        return memory

    def __init__(self):
        super().__init__(
            memory=SimpleChatMemory(
                purpose="You're a robot that writes poems from the user's input."
            ),
            pre_call_chain=[
                self.create_log,
                self.remember_user_interaction,
                self.create_instructions,
            ],
            post_call_chain=[self.log_what_I_did, self.stop_the_robot],
            ai_model=OpenAIChatCompletion(),
        )

if __name__ == "__main__":
    poetry_robot = PoetryRobot()
    some_input_might_be = ChatMessage(
        role="user",
        content="I'm a user input. I heard that no matter what I say, you'll write a poem about it?",
    )
    result = poetry_robot.process(some_input_might_be)
```