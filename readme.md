
# RobAI Framework Documentation

---

## Introduction
RobAI is a _simple_ but powerful framework designed to make working with AI models more intuitive. It is 'memory' oriented, with a simple flow: it calls all the pre-call functions, it then calls the AI model, then it calls all the post-call functions. 

The common object at every step of the journey is the `memory` object, of type `BaseMemory`. The memory is always available on the robot at `robot.memory`

That's it. When the robot is finished, it returns its memory object.

Memory is just a pydantic class where you can store anything the robot might need to 'do' whatever it's tasked with. Robot's need a `purpose`, and as you might have guessed, the purpose of the robot is stored in the robot's memory at `robot.memory.purpose`.

You might imagine this as the robot's 'system prompt', it's what the robot is told it's purpose will be when it's initialised. Have a look at the 'AIRobot' init method and you'll see that the robot's pupose is added as an initial 'system' message to it's message history. 

The framework has been written so that writing code for large *language* models feel closer to writing *language*. Writing AI code should feel intuitive, it should be rooted in concepts familiar to humans, and the code should read like a 'real' interaction. For things to feel familiar, we have to know exactly what happens when we call process on our robot at `robot.process(some_input_string_or_model)`

## What exactly happens when you call the robot?

When you've finished making your robot, you'll call .process() on the robot and this is _exactly_ what will happen.
1. Developer calls `robot.process(input)`
2. `input` is added to the robot's `memory` object at `memory.input_model`
3. `memory` is passed from function to function in the `pre_call_chain`
4. `memory` is then sent to `ai_model` which parses the `memory.instructions_for_ai` attribute, which is always a list of `ChatMessage(role='foo' content='this is basically the prompt)` objects. 
5. `robot.ai_model` then sends those parsed instructions to the AI model, and puts the response in `memory.ai_response`
6. Robot passes the `memory` object (with a new `ai_response`) to every function in `post_call_chain`
6. Robot returns the `memory` object
    - If `memory.set_complete()` is called somewhere in the chain (usually post-call), the `memory` object is returned
    - If `memory` is NOT complete, the `memory` is passed from `post_call_chain` to `pre_call_chain` again and it keeps going until something triggers `memory.set_complete()` in the chain.

As simple as this is, it's actually a very powerful and flexible setup. Robots can easily be chained together in the pre-call and post-call chains because you can rely on the fact that `memory.instructions_for_ai` will *always* be `List[ChatMessage]`. Whatever you're doing with a large language model, you can certainly 'do it' via the medium of ChatMessage objects. Standardising this drastically reduces complexity.

When developing with Robai, you only need to use the `pre-call` functions to create the `memory.instructions_for_ai` for the AI model at step 4. In the `post-call` functions, you can chain the robot to another robot, process the response further, or even send the robot back to `pre-call` if the AI response is not as expected. As soon as your robot passes some test, which you set in post-call, just call `memory.set_complete()` and the robot will return the entire memory object. Other than that, you can really do whatever you like in pre-call or post-call chains.

# A complete example
## Summarising text longer than the context window allows

```python
from robai.memory import BaseMemory
from robai.base import AIRobot

# Memory only needs to have an input_model defined, which is what is passed to the robot
# When `robot.process(input_model)` is called.
class SummaryRobotMemory(BaseMemory):
    input_model: str = None
    chunks: List[str] = []
    # The context window for gpt3.5 is now 16,000 tokens. Each chunk must be less than that.
    # Assumes the model will summarise 10,000 tokens to 6,000 tokens.
    chunk_length_limit: int = 10000
    summaries: List[str] = []
    current_chunk_index: int = 0
    total_chunks: int = 0
    instructions_for_ai: List[ChatMessage] = None

class SummaryRobot(AIRobot):
    def __init__(self):
        super().__init__()
        self.ai_model: OpenAIChatCompletion = OpenAIChatCompletion()
        self.memory: SummaryRobotMemory = SummaryRobotMemory(
            purpose="""
                    Please extract all key facts of this text into a format similar to shorthand
                    but human readable.
                   """.strip(
                "\n"
            ),
        )
        self.pre_call_chain: List[Callable] = [
            self.split_text_into_chunks,
            self.provide_context_for_chunk,
        ]
        self.post_call_chain: List[Callable] = [
            self.append_summary_and_check_complete,
        ]
    """
    PRE CALL FUNCTIONS
    """
    # PRE-CALL 1
    def split_text_into_chunks(
        self,
        memory: SummaryRobotMemory,
    ) -> SummaryRobotMemory:
        to_summarise = memory.input_model
        chunk_length_limit = memory.chunk_length_limit
        if not memory.chunks:
            chunks = self.split_text_into_token_chunks(to_summarise, chunk_length_limit)
            memory.chunks = chunks
            memory.current_chunk_index = 0
            memory.total_chunks = len(chunks)
        return memory

    # PRE-CALL 2
    def provide_context_for_chunk(
        self,
        memory: SummaryRobotMemory,
    ) -> SummaryRobotMemory:
        # Pop the first chunk as the current content
        to_summarize = memory.chunks.pop(0)
        current_chunk_num = memory.current_chunk_index + 1
        context = f"""
        You are summarising text. You are at section {current_chunk_num} of {memory.total_chunks}. \n\n
        Here's what you've summarised so far {memory.summaries}\n\n
        You must now summarise this chunk of text and we'll add it to the summary so far: {to_summarize}\n\n
        """
        # the instructions for the AI model are always in the form of ChatMessages.
        memory.instructions_for_ai = [ChatMessage(role="user", content=context)]
        return memory

    """
    POST CALL FUNCTIONS
    """
    # POST-CALL 1
    def append_summary_and_check_complete(
        self,
        memory: SummaryRobotMemory,
    ) -> SummaryRobotMemory:
        if not hasattr(memory, "summaries"):
            memory.summaries = []
        # APPEND THE SUMMARY WE HAVE RECEIVED TO THE LIST OF SUMMARIES
        memory.summaries.append(memory.ai_response.content)
        memory.current_chunk_index += 1
        # CHECK IF WE ARE DONE
        if memory.current_chunk_index == memory.total_chunks:
            # There are no more chunks to summarise,
            # we are done. memory.set_complete() means the robot will not go back to pre-call.
            memory.set_complete()
        else:
            # There are more chunks to summarise. We are not done.
            # The robot sends everything back to pre-call and
            # we'll summarise the next chunk.
            # Note that in pre-call we pop() the chunks, so when they're processed they are gone.
            pass

        return memory

    # UTILITY
    def split_text_into_token_chunks(
        self, text: str, chunk_length_limit: int
    ) -> List[str]:
        """
        Split the text into chunks based on an estimated token count.
        """
        average_tokens_per_word = 1.5
        words = text.split()
        words_per_chunk = int(chunk_length_limit / average_tokens_per_word)

        chunks = []
        for i in range(0, len(words), words_per_chunk):
            chunks.append(" ".join(words[i : i + words_per_chunk]))

        return chunks


text_to_summarise = """current state-of-the-art in propulsion, material science, energy production and storage.
        The knowledge we stand to gain should spur us toward a more enlightened and sustainable future,
        one where collective curiosity is ignited, and global cooperation becomes the norm, rather than the
        exception.
        Thank You ."""
input_model: str = text_to_summarise
memory = robot.process(input_model)
robot.console.print(memory.ai_response)

```

### Installation
Installation Instructions:
1. Using pip:
For users who prefer pip, they can install your package directly from GitHub (once you push your changes) with:

```
pip install git+https://github.com/philmade/robai.git
```


2. Using poetry:

```
git clone https://github.com/philmade/robai.git
cd robai
poetry install
```

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
