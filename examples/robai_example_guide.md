
### Building a Robot using ROBAI

---

**Introduction:**

We'll start by building a basic robot that uses a fake AI model. This will serve as our foundation. Later, we'll integrate a real AI model and add more capabilities. Throughout this guide, we'll progressively add complexity to our robot, explaining each step.

---

**Import Required Modules:**

```python
from robai.memory import BaseMemory
from robai.languagemodels import FakeAICompletion
from robai.examples.example_utility import FakeDB
from robai.base import AIRobot
from pydantic import BaseModel
from typing import List, Callable
from pprint import pprint
```

---

**Example Implementation of a Robot:**

Building a robot involves several steps. These include setting up the robot, defining its input, memory, pre-call chain, post-call chain, and then running it. 

---

**Ground Zero - The Robot:**

The `AIRobot` class provides the basic structure of a robot. It requires:

1. A `robot.ai_model` which is the AI component. It could be any AI model like GPT-3, Llama, etc.
2. A `robot.memory` which stores essential data for the robot.
3. A `robot.pre_call_chain` which is a list of functions executed before calling the AI model.
4. A `robot.post_call_chain` which is a list of functions executed after the AI model returns a result.

---

**Define the Robot's Input Model:**

The input model serves as the robot's "inbox". It contains data passed to the robot when it's invoked.

```python
class InputClothesBasket(BaseModel):
    users_question: str
    clothing_items: List[str]
    clothing_costs: List[float]
```

---

**Define the Robot's Memory:**

The robot's memory is where it stores all the data it needs to perform its tasks. It must have an input_model, and here we set it to the input model we defined above.

```python
class RobotMemoryClothes(BaseMemory):
    input_model: InputClothesBasket = None
    instructions_for_ai: List[ChatMessage] = None
    recommended_clothes: List[str] = None
```

---

**Define the Pre-Call Chain:**

The pre-call chain consists of functions that the robot will execute before it calls the AI model. Each function must take one argument and return one argument - the robot's memory that we have just defined.

```python
def query_recommended_clothes(memory: RobotMemoryClothes) -> RobotMemoryClothes:
    # When the robot is called, the memory object will have the input_model set to whatever the input was.
    # So we can now access that data.
    clothing_items = memory.input_model.clothing_items

    # Now we can add anything we need to memory that will help the robot do its job
    # Like querying a database based on input and getting relevant data.
    memory.recommended_clothes = FakeDB.similar_clothes(clothing_items)
    # We have added a DB query to the memory and returned it.
    # The memory will now be sent to the next function in the chain.
    return memory


def add_message_to_memory(memory: RobotMemoryClothes) -> RobotMemoryClothes:
    # If we want to keep a 'chat like' history of the interaction, we can add messages to memory
    # The ROBAI framework is about imagining how to interact with AI.
    # We can imagine that an AI would make good use of remembering what the user/robot has said, so let's add it to memory.
    message = ChatMessage(
        role="user",
        content=f"The user's clothes basket is: {memory.input_model.dict()}",
    )
    # All memory objects have a default method called add_message, which adds a ChatMessage to memory
    memory.add_message(message)
    return memory


def finally_create_style_instructions(memory: RobotMemoryClothes) -> RobotMemoryClothes:
    """
    The point of the pre-call chain is to prepare a prompt / question for the AI model. Our final function should do that, so
    that's what we're doing here. We're creating a prompt for the AI model to use.
    """
    instructions = f"""
    A user has {memory.input_model.clothing_items} in her shopping basket. 
    The database has recommended {memory.recommended_clothes} as similar items.
    The user has a budget of {sum(memory.input_model.clothing_costs)}.
    The user has a question about their shopping basket: '{memory.input_model.users_question}' 
    Can you answer her question and help to style her basket, to her budget, with the recommended items?
    Here's your interaction history with the user: {[message.dict() for message in memory.message_history]}
    """
    # We have created an f-string that contains all the information we need to pass to the AI model
    # But remember we set memory.instructions_for_ai as type List[ChatMessage] above?
    # Hover over memory.instructions_for_ai above and you'll see that it's type is List[ChatMessage]
    memory.instructions_for_ai
    #
    # So this is what we'll set as the instructions.

    memory.instructions_for_ai[ChatMessage(role="user", content=instructions)]
    return memory


# Here is our pre-call chain. It's simply a list of functions that will be called in order.
pre_call_recommend_and_style = [
    query_recommended_clothes,
    add_message_to_memory,
    finally_create_style_instructions,
]

```

---

**Define the Post-Call Chain:**

The post-call chain consists of functions that the robot will execute after the AI model returns a result. They too take and return the memory object. Here, in postcall, we actually have the response from the ai_model available in memory. It's now available at memory.ai_response, memory.ai_string_response, and memory.ai_raw_response. You can use these responses to do whatever you want. However, when you're done, the postcall chain MUST call memory.set_complete() or the robot sends the memory back to pre-call and the process starts again

```python

def stop_the_robot(memory: RobotMemoryClothes) -> RobotMemoryClothes:
    # In our case, we don't want to do anything in post_call, so we just stop the process
    # We do this by calling memory.set_complete()
    # WITHOUT THIS THE ROBOT WILL NEVER STOP!
    memory.set_complete()
    return memory


post_call_do_nothing = [stop_the_robot]
```

---

**Construct the Robot:**

Now, with all the components in place, we can construct the robot.

```python
class ClothingRobot(AIRobot):
    ai_model: FakeAICompletion
    pre_call_chain: List[Callable] = pre_call_recommend_and_style
    post_call_chain: List[Callable] = post_call_do_nothing
    memory: RobotMemoryClothes

    # WHEN BUILDING A ROBOT WE HAVE TO CALL SUPER() AND PASS IN THE MEMORY CLASS
    def __init__(self):
        super().__init__(
            memory=RobotMemoryClothes(
                purpose="You are a clothing robot, you will recieve clothing items and queries from a user, and will return a style"
            ),
            pre_call_chain=pre_call_recommend_and_style,
            post_call_chain=post_call_do_nothing,
            ai_model=FakeAICompletion(),
        )
```

---

**Run the Robot:**

Finally, we can instantiate and run the robot.

```python
clothing_robot = ClothingRobot()

robot_input_model = InputClothesBasket(
    users_question="What should I wear?",
    clothing_items=["blue jeans", "black t-shirt", "red jacket"],
    clothing_costs=[10.0, 5.0, 15.0],
)

output = clothing_robot.process(robot_input_model)
pprint(output.dict())
```
