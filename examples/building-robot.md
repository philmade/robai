Usage:
```python
from robai.base import AIRobot

ai_robot = AIRobot()
```

```python
class SimpleChatMemory(BaseMemory):
    purpose: str = "Chat with a human"
    input_model: ChatMessage = None

```
Don't overthink the memory, just know that you can add whatever you like to it, and it's very helpful. You must only set a purpose. The rest of that memory we'll come to later. 

Ok, so we have looked at our robot's memory - now what?

There's the pre-call chain, which is just a list of functions that are called _before_ the AI model is actually called. You can write whatever you want in these functions, as long as each function takes a single argument - the memory object - and each function returns a single object - the memory object. 

It's memory oriented... see?

The only thing that you must ensure is that the final function in your pre-call chain sets the 'instructions_for_ai' on the memory object. Have a look at it up there in the memory object. Everything before that, you can do _whatever you want_ as long as those instructions are set somehow. Without instructions, the ai_model doesn't have anything to do!

So now we have our AI instructions - easy. Now what?

Now the memory object is passed to the ai_model, and we call() it, and what argument do we use in the call()? The memory object! 

The call() method takes the memory object, and passes the memory.instructions_for_ai as the prompt to the language model. As such, memory.instructions_for_ai should be json serializable. The AI Model then returns.... the memory object! Exactly _how_ the ai_model returns its response is up to you, because you can implement the ai_model however you like. Here's an intuitive way to return the repsponse for 'chat completions' via OpenAI:

```python
class FakeAICompletion(BaseAIModel):
    is_ready: bool = False

    def setup(self, *args, **kwargs):
        self.is_ready = True

    def call(self, memory: BaseMemory) -> BaseMemory:
        if not self.is_ready:
            raise Exception("Model is not ready - YOU MUST CALL SETUP")
        # For the purposes of this FakeAICompletion, we use Faker, but in a real LLM you'd call language model 
        fake_response = call_any_ai_somehow(prompt=memory.instructions_for_ai)
        # Here fake sentences would be the string response from the AI.
        message = ChatMessage(role="assistant", content=fake_response)
        # We call a memory function and now the AI's response is stored, in memory, as a ChatMessage. Done.
        memory.add_message(message)
        # And now we return the memory object!
        return memory
```

Now,

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
