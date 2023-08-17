from robai.memory import BaseMemory
from pydantic import BaseModel
from typing import List
from pprint import pprint

# =========================================
# EXAMPLE IMPLEMENTATION
# =========================================

# =========================================
# 1. What 'goes into' your robot?
# =========================================


class ClothesBasket(BaseModel):
    clothing_items: List[str]
    clothing_costs: List[float]


# =========================================
# 1.1 What 'comes out of' your robot?
# Use a ChatMessage for now.
# =========================================

from robai.in_out import ChatMessage


# =========================================
# 2. Define your memory
# =========================================
class RobotMemoryClothes(BaseMemory):
    # THE ROBOT NEEDS THESE TWO THINGS
    # input model is what goes into your robot
    # Output model is what you're getting back from the AI
    # =========================================
    input_model: ClothesBasket = None  # we set a default so no need to give instances immediately. Will make sense later.
    # =========================================

    # Now you can add whatever your robot might need to do its job
    recommended_clothes: list[str] = None


# Simple fake DB
class FakeDB:
    def similar_clothes(query: str):
        return ["dark denim jeans", "Aztec print jacket", "white t-shirt"]

    def store_the_db(thing: dict):
        return "Success"


# =========================================
# 3. Define your PreCall chain
# =========================================


def query_recommended_clothes(memory: RobotMemoryClothes) -> RobotMemoryClothes:
    users_clothes = memory.input_model  # <--- Check the ClothesMemory class above
    memory.recommended_clothes = FakeDB.similar_clothes(users_clothes.clothing_items)
    return memory


def add_message_to_memory(memory: RobotMemoryClothes) -> RobotMemoryClothes:
    # We don't need to do this in this example, but this is how you'd add a ChatMessage message to memory
    message = ChatMessage(
        role="user",
        content=f"The user's clothes basket is: {memory.input_model.dict()}",
    )
    memory.add_message(message)
    return memory


def create_style_instructions(memory: RobotMemoryClothes) -> RobotMemoryClothes:
    instructions = f"""
    A user has {memory.input_model.clothing_items} in her shopping basket. 
    The system has recommended {memory.recommended_clothes} as similar items.
    The user has a budget of {sum(memory.input_model.clothing_costs)}.
    Can you recommend a style for the user based on the items in her basket, her budget, and the recommended items?
    Here's your interaction history with the user: {[message.dict() for message in memory.message_history]}
    """
    memory.instructions_for_ai = instructions
    # Again, the message history may not be needed in this example, but it's there if you need it. message_history is a list of ChatMessage objects
    return memory


pre_call_recommend_and_style = [
    query_recommended_clothes,
    add_message_to_memory,
    create_style_instructions,
]


# =========================================
# 3. Define your PostCall chain
# =========================================
def stop_the_robot(memory: RobotMemoryClothes):
    memory.set_complete()  # <--- When memory.set_complete() is done, the robot is finished
    return memory


post_call_do_nothing = [stop_the_robot]


# =========================================
# 4. Define your robot
# =========================================

from robai.base import AIRobot
from robai.llm import FakeAICompletion

clothing_recommender = AIRobot(
    memory=RobotMemoryClothes(
        purpose="You will recieve clothing selections from customers, with relevant information about their budget and preferences. Your job is to style the recommended clothing outfits"
    ),
    pre_call_chain=pre_call_recommend_and_style,
    post_call_chain=post_call_do_nothing,
    ai_model=FakeAICompletion,
)

# Have to setup the AI model
clothing_recommender.ai_model.setup()  # <--- this is where we'd pass parameters to the AI model, temperature, max_tokens etc


# =========================================
# 5. Run your robot
# =========================================

# Create some fake data
fake_clothes = ClothesBasket(
    clothing_items=["blue jeans", "black t-shirt", "red jacket"],
    clothing_costs=[10.0, 5.0, 15.0],
)

# This is how I want to use it - pass an instance of memory.input_model (ClothesBasket) to the robot andn then check it's the correct type

# Run the robot
output = clothing_recommender.process(fake_clothes)
# BASIC FLOW
# memory passes through all functions in pre-call
# memory passess to call() method on ai_model, using memory.instructions_for_ai as the prompt
# memory.add_message(role='ai', content='the actual ai response') is called by the ai_model
# memory passes through all functions in post-call
# memory.set_complete() is called by some function in post-call
# memory is returned
# if not memory.complete:
# memory passes through all functions in pre-call and the loop continues.

pprint(output.dict())


# MORE DETAILED FLOW
# 1) The robot checked that the type of argument passed to process() matched the type defined in memory.input_model
# 2) The robot added the argument passed to process() to memory.input_model
# 3) The robot sequentially passed the memory to each function in the pre_call_chain. Each time, the memory was modified and passed to next function.
# 4) After pre-call chain finishes, the robot passes the memory object to the AI model's call() argument.
# 5) Using memory.instructions_for_ai for the prompt, the call() function calls the AI Model adding its response as a ChatMessage to memory.message_history
# 6) The robot passes memory to ran all the functions in the post_call_chain, recieving and passing altered memory object each time
# 7) When a function in the post_call_chain calls memory.set_complete(), the robot stops running and returns the memory object
# 8) If memory.set_complete() is not called, the robot passes the memory back to pre-call-chain. This loop continues until memory.set_complete() is called
