
# RobAI Framework Documentation

---

## Introduction
RobAI is a powerful framework designed to streamline the construction and utilization of AI robots. By following the steps outlined below, developers can easily integrate AI functionalities into their applications and services.

## 1. Define a Custom Memory

Every AI robot needs a memory to store input, intermediate results, and final output. Start by defining a custom memory that inherits from `BaseMemory`.

For example, if you're building a translation robot:

```python
from robai.memory import BaseMemory
from pydantic import BaseModel

# Define the input model
class TranslationRequest(BaseModel):
    source_language: str
    target_language: str
    content: str

# Define the custom memory class
class TranslationMemory(BaseMemory):
    input_model: TranslationRequest
    instructions_for_ai: str = ""
    translated_content: str = ""
```

## 2. Create the Pre-call Chain

Before the AI processes the data, you can define a series of functions to manipulate or prepare the data. These functions should be added to the pre-call chain.

For our translation example:

```python
async def set_translation_instructions(memory: TranslationMemory) -> TranslationMemory:
    memory.instructions_for_ai = f"Translate the following from {memory.input_model.source_language} to {memory.input_model.target_language}: {memory.input_model.content}"
    return memory
```

## 3. Create the Post-call Chain

After the AI processes the data, you can define functions to further process or manipulate the AI's output. These functions should be added to the post-call chain.

Continuing with the translation example:

```python
async def set_translated_content(memory: TranslationMemory) -> TranslationMemory:
    memory.translated_content = memory.output
    return memory
```

## 4. Construct the Robot

With your memory and function chains defined, construct your robot using the `AIRobot` class.

```python
from robai.base import AIRobot
from robai.llm import FakeAICompletion

translator_robot = AIRobot(
    memory=TranslationMemory(purpose="Language Translation"),
    pre_call_chain=[set_translation_instructions],
    post_call_chain=[set_translated_content],
    ai_model=FakeAICompletion
)
```

## 5. Run the Robot

Execute your robot using the provided data:

```python
translation_request = TranslationRequest(source_language="English", target_language="Spanish", content="Hello, World!")
result = await translator_robot.aprocess(input_data=translation_request, stream=False)
print(result.translated_content)
```

---

## Conclusion

The RobAI framework offers a structured approach to building AI robots. By defining custom memories, pre-call and post-call chains, and integrating with AI models, developers can easily create powerful AI solutions tailored to their specific needs.

Feel free to explore and extend the framework to suit more complex requirements or integrate with other AI models. Happy coding!
