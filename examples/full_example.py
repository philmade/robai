from robai.memory import BaseMemory, SimpleChatMemory
from robai.languagemodels import FakeAICompletion
from robai.utility import FakeDB
from robai.base import AIRobot
from robai.in_out import ChatMessage
from pydantic import BaseModel
from typing import List, Callable
from robai.utility import pprint_color

# =========================================
# EXAMPLE IMPLEMENTATION OF A ROBOT
# =========================================
"""
We're going to build a robot that takes a user's shopping basket and recommends a
style based on the items in the basket, the user's budget, and the recommended items.

To help us have an intuitive understanding of what's going on, we're going to develop this robot in a slightly different way.
We'll start with the robot itself, because this way we'll see exactly what a robot needs and how it works.

Then we'll define what should go 'into' the robot, this is usually the user's input.
Then we'll define some memory where we'll store what the robot will need for its job
Then we'll define some functions for the robot to call BEFORE it calls the AI model, this is the PRE-call chain
Then we'll define some functions for the robot to call AFTER it calls the AI model, this is the POST-call chain
Finally, we'll set a condition where the robot's memory is set to complete, and the robot is finished.
The robot will then return the memory.

"""

# =========================================
# GROUND ZERO - THE ROBOT
# =========================================

"""
Starting at the AIRobot class helps us understand exactly what's going on.
An AIRobot needs:
    1) A robot.ai_model - this is the AI bit, and it could be LLama, GPT3, or anything else. In this case, we'll use a fake AI model.
    2) A robot.memory - this is where the robot will store what it needs to do its job. It has two main parts.
    3) A robot.pre_call_chain - this is a list of functions that will be called before the AI model is called.
    4) A robot.post_call_chain - this is a list of functions that will be called after the AI model is called.
"""

# We're setting up the robot now so we get helpful type hints to guide us along the way.


class ClothingRobot(AIRobot):
    ai_model: FakeAICompletion  # <--- This is a subclass of BaseAIModel. You can make your own very easily or get one from robai.languagemodels
    memory: BaseMemory  # <--- We shouldn't use BaseMemory, we should use our own memory class, which we're about to define
    pre_call_chain: List[Callable]  # <--- a list of functions
    post_call_chain: List[Callable]  # <--- a list of functions

    # WHEN BUILDING A ROBOT WE HAVE TO CALL SUPER() AND PASS IN THE MEMORY CLASS
    def __init__(self):
        super().__init__(
            # These are placeholders for now, we'll develop these properly later
            memory=SimpleChatMemory(
                purpose="You are a clothing robot, you will recieve clothing items and queries from a user, and will return a style"
            ),
            pre_call_chain=[],
            ai_model=FakeAICompletion(),
        )


# We now have a robot! Great! But it doesn't do anything yet, but it can give us useful information about what we need to do next.
clothing_robot = ClothingRobot()

# The ai_model is now available on the robot at robot.ai_model. The memory is at robot.memory
# Pay attention to robot.ai_model.instructions_for_ai.
# Hover over it on vscode and you'll see it's type is List[ChatMessage]
clothing_robot.ai_model.instructions_for_ai

# THIS IS ALL WE NEED TO REMEMBER.
# When we're coding our robot, all we need to do ensure the robot.memory.instructions_for_ai is of type List[ChatMessage]
# Everything else is entirely flexible.
# So let's get started!


# =========================================
# 1. DEFINE YOUR ROBOTS INPUT MODEL
# 1.1 What 'goes into' your robot?
# =========================================
"""
Input Models for the robots are simply Pydantic Models, they can contain anything you like. Imagine these as the data that's 
passed to your robot when it's called. It's your robot's 'inbox', which it reads, and then does stuff.
All your robot's interactions start here!
Type annotations should always be set, because they help the framework check that your robot will work.
Even though this looks very structured, we have space here for a natural language input in the 'users_question' of type <str>
"""


class InputClothesBasket(BaseModel):
    users_question: str
    clothing_items: List[str]
    clothing_costs: List[float]


# =========================================
# 2. DEFINE YOUR MEMORY
# 2.1) Inherit from robai.BaseMemory
# =========================================
"""
    The memory is where your robot will store what it needs to do its job. 
    It NEEDS an input_model, and a type annotation for the instructions_for_ai.

    # 2) robot.memory.input_model is what how we interact with the robot. We just defined it above.
    # 2.1) robot.memory.instructions_for_ai is what the robot will send to the AI model
    # 2.2) Remember we already looked at robot.ai_model.instructions_for_ai_type above? Different AI models require different input types.
    # 2.3) SO, the type annotation on robot.memory.instructions_for_ai must match the input type annotation of the robot.ai_model.instructions_for_ai
"""


class RobotMemoryClothes(BaseMemory):
    input_model: InputClothesBasket = (
        None  # we set a default so no need to give instances immediately
    )
    instructions_for_ai: List[
        ChatMessage
    ] = None  # This must match what's at robot.ai_model.instructions_for_ai Again, set a default of None.

    # WHAT OTHER INFORMATION DOES YOUR ROBOT NEED?
    """
    # Now you can add whatever additional memory attributes your robot might need to do its job
    # You can revisit this later, adding and changing as you need.
    # You can add whatever you like here.
    """
    recommended_clothes: List[str] = None


# =========================================
# 3. DEFINE YOUR PRE-CALL CHAIN
# =========================================
"""
    # 3.1) These are all the functions that will be called before the robot.ai_model is called
    # 3.2) All functions in precall must take and return a single argument, the memory we just defined above.
"""


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
    memory.add_message_to_history(message)
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
    Here's your interaction history with the user: {[message.dict() for message in memory.message_history if message]}
    """
    # We have created an f-string that contains all the information we need to pass to the AI model
    # But remember we set memory.instructions_for_ai as type List[ChatMessage] above?
    # Hover over memory.instructions_for_ai above and you'll see that it's type is List[ChatMessage]
    # memory.instructions_for_ai
    #
    # So this is what we'll set as the instructions.

    memory.instructions_for_ai = [ChatMessage(role="user", content=instructions).dict()]
    return memory


# Here is our pre-call chain. It's simply a list of functions that will be called in order.
pre_call_recommend_and_style = [
    query_recommended_clothes,
    add_message_to_memory,
    finally_create_style_instructions,
]


# =========================================
# 4. Define your PostCall chain
# =========================================
"""
    # 4.1) The robot has now called robot.ai_memory.call() and has recieved a response from the AI model.
    # 4.2) The ai_model always puts the response in robot.memory.ai_response, memory.ai_string_response, and memory.ai_raw_response.
    # 4.3) You can now use these responses to do whatever you want.

"""


def stop_the_robot(memory: RobotMemoryClothes) -> RobotMemoryClothes:
    # In our case, we don't want to do anything in post_call, so we just stop the process
    # We do this by calling memory.set_complete()
    # WITHOUT THIS THE ROBOT WILL NEVER STOP!
    memory.set_complete()
    return memory


post_call_do_nothing = [stop_the_robot]


# =========================================
# 5. LETS RECONSTURCT THE ROBOT WITH WHAT WE BUILT
# =========================================


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


# =========================================
# 6. Run your robot
# =========================================

# Make an instance of your robot

if __name__ == "__main__":
    clothing_robot = ClothingRobot()

    # Create an input model instance of type clothing_robot.memory.input_model
    robot_input_model = InputClothesBasket(
        users_question="What should I wear?",
        clothing_items=["blue jeans", "black t-shirt", "red jacket"],
        clothing_costs=[10.0, 5.0, 15.0],
    )

    # Process the input model
    output = clothing_robot.process(robot_input_model)
    intro_ = """=========================================
THE MEMORY FROM THE CLOTHING ROBOT
========================================="""
    pprint_color(intro_)

    pprint_color(output.dict())

    next_ = """=========================================
THE STRING RESPONSE FROM THE AI: 
memory.ai_string_response
========================================="""

    pprint_color(next_)

    pprint_color(output.ai_string_response)

    final_ = """=========================================
HERE iS WHAT THE ROBOT DID
memory passed through all functions in pre-call
memory passed to call() method on ai_model, using memory.instructions_for_ai as the prompt
memory.add_message(role='ai', content='the actual ai response') is called by the ai_model
memory passes through all functions in post-call
memory.set_complete() is called by some function in post-call
memory is returned
if not memory.complete:
memory passes through all functions in pre-call and the loop continues.
========================================="""
    pprint_color(final_)
