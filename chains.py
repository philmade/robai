from .languagemodels import BaseMemory
from .in_out import ChatMessage, AIMessage


# =========================================
# THE SIMPLEST PRE CALL CHAIN
# =========================================
def create_instructions(memory: BaseMemory) -> BaseMemory:
    # Assuming the input model is of type ChatMessage, we'll add it to memory
    # add_message is a default method of the BaseMemory class
    memory.add_message(memory.input_model)

    # Set the instructions for the AI to be the message the robot recieved from the user, which will always be at memory.input_model
    # Have a look at the ChatMessage pydantic object, it's dict form is perfectly formatted as a 'ChatMessage' for the AI
    memory.instructions_for_ai = memory.input_model.dict()

    # Return the memory
    return memory


# =========================================
# THE SIMPLEST POST CALL CHAIN
# =========================================
def stop_the_robot(memory: BaseMemory) -> BaseMemory:
    memory.set_complete()  # <--- When memory.set_complete() is done, the robot is finished
    return memory


do_nothing_in_post_call = [stop_the_robot]


# =========================================
# SETTING IT UP EXAMPLE
# =========================================

# from robai.base import AIRobot
# from robai.llm import FakeAICompletion

# simple_robot = AIRobot(
#     memory_class=BaseMemory,
#     pre_call_chain=[create_instructions],
#     post_call_chain=[do_nothing_in_post_call],
#     ai_model=FakeAICompletion,
# )

# Have to setup the AI model
# simple_robot.ai_model.setup()  # <--- this is where we'd pass parameters to the AI model, temperature, max_tokens etc


# =========================================
# RUNNING IT
# =========================================

# Create the input data
# message = ChatMessage(
#     role='user',
#     content='Hello, I am a user'
# )

# Run the robot
# output = simple_robot.process(message)
# The output will be of type BaseMemory because that's what post_call_chain returns
