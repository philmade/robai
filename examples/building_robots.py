from robai.memory import SimpleChatMemory
from robai.languagemodels import OpenAIChatCompletion
from robai.base import AIRobot
from robai.in_out import ChatMessage
from pydantic import BaseModel
from typing import List, Callable
from loguru import logger


# =========================================
# HOW TO BUILD A ROBOT
# =========================================
# 1. INHERIT FROM AIRobot
class PoetryRobot(AIRobot):
    # ==========
    # 2. HERE'S WHAT YOU MUST BUILD FOR YOUR ROBOT
    # ==========
    ai_model: OpenAIChatCompletion  # <--- This is a subclass of BaseAIModel.  # You can make your own very easily or get one from robai.languagemodels
    memory: SimpleChatMemory  # <--- Start with SimpleChatMemory, # You can change this later
    input_model: ChatMessage  # <--- This is the input model for your robot. It's a ChatMessage by default, but you can change it to anything you like
    purpose: str = "I think you're a robot that writes poems from the user's input."
    pre_call_chain: List[Callable]  # <--- a list of functions
    post_call_chain: List[Callable]  # <--- a list of functions

    # =========================================
    # 3 This is a good place to start - what should your robot receive as input?
    # =========================================
    # Your robot can accept any subclass of BaseModel.
    # To keep things simple, we'll start with a ChatMessage
    # We're redefining ChatMessage to be explicit for demonstration purposes,
    # but you could just import this from robai.in_out import ChatMessage
    # or define it in your own module and import it as MyRobotInputModel.
    class ChatMessage(BaseModel):
        role: str
        content: str

    # =========================================
    # 4. DEFINE SOME PRE-CALL FUNCTIONS
    # =========================================
    # Now we can start to customise the robot to do whatever we want.
    # Pre-call and post-call functions are just functions. We will later register them to their correct place.
    # For now, just imagine them as 'things the robot will do before calling the AI'.
    # Your robot will call all pre-call functions, then call the ai_model, then call post-call functions
    # The pre-call and post-call functions must take in a memory object and return a memory object
    # The memory object should be of the same type PoetryRobot.memory you annotated above
    # We can have as many functions in pre_call/post-call as needed
    # You don't need to define these functions ON the robot itself, but it's good practice,
    # because it gives you access to all robot functions in pre and post-call chains.

    # =========================================
    # 4.1 SIMPLE LOGS BEFORE THE ROBOT CALLS THE AI [OPTIONAL]
    # =========================================
    def create_log(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        self.printer.pprint_color(
            f"Robot {self.__class__.__name__} has just received some input, which is: {memory.input_model}".strip()
        )
        return memory

    # =========================================
    # 4.2 ADDING INTERACTION HISTORY TO MEMORY [OPTIONAL]
    # =========================================
    # This is a useful thing to do in pre-call, because these messages can be used for context later
    # The 'add_message_to_history' method is available on all memory objects,
    # We can call it in a pre-call function like this:

    def remember_user_interaction(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        memory.add_message_to_history(memory.input_model)

        # It should be noted that add_message_to_history requires a ChatMessage,
        # even if your input model is not that...
        # Our user input_model can actually be any structure, not always a ChatMessage.

        # So if input_model were not defined as type ChatMessage, we could do this:
        memory.add_message_to_history(
            ChatMessage(role="user", content=memory.input_model.query_string)
        )
        # Or this:
        memory.add_message_to_history(
            ChatMessage(role="user", content=str(memory.input_model.dict()))
        )

        return memory

    # =========================================
    # 4.3 CREATING INSTRUCTIONS/PROMPT FOR THE AI
    # =========================================
    # The whole purpose of the pre-call chain is to actually sets some instructions for the AI
    # Obviously, the instructions should be relevant to the robot's purpose!
    def create_instructions(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        # What do we have to work with in pre-call?
        # Have a look at the 'SimpleChatMemory' class, and the 'BaseAIMemory' it inherits from
        # All memory has a 'message_history' attribute, which is a list of ChatMessages
        # It also has an .instructions_for_ai attribute, which must _always_ be a List[ChatMessages]
        # Whatever your robot's task, it must be 'explained' in a list of ChatMessages.

        memory.instructions_for_ai = [memory.system_prompt, memory.input_model]
        # The line above shows the simplest set of instructions possible.
        # The 'system_prompt' is available in memory and it will always be:
        #   ChatMessage(role='system', content='The Purpose of the Robot')
        # It is also added to message_history on init, so you could also get it like this:
        # system_prompt = memory.message_history[0]

        # So here, our instructions for the AI are the system_prompt message, plus the input_model instance,
        # populated with the user input

        # It's equivalent to this:
        memory.instructions_for_ai = [
            ChatMessage(role="system", content=self.purpose),
            ChatMessage(
                role="user",
                content="Whatever the user said when the robot was called, like, 'Hey, I heard you're great at writing poems?!",
            ),
        ]
        # But let's set the instructions back to the dynamically created values.
        memory.instructions_for_ai = [memory.system_prompt, memory.input_model]
        logger.info(
            f"Hi, I'm Robot {self.__class__.__name__} and I'm about to call the AI with instructions: {memory.instructions_for_ai}"
        )
        return memory

    # =========================================
    # 5. DEFINE SOME POST-CALL FUNCTIONS
    # =========================================
    # What should your robot do after it has called the AI?
    # You can have as many post-call functions as you like.

    def log_what_I_did(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        logger.info(
            f"""I'm robot {self.__class__.__name__} and I just called the AI with instructions: {memory.instructions_for_ai}
            I received the following response: {memory.ai_response}
            That response is available in my memory module at robot.memory.ai_response. 
            It's in the form of a ChatMessage
            The raw response from the language model is available at robot.memory.ai_raw_response
            """
        )

    def stop_the_robot(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        # In our case, we don't want to do much else in post_call, so we just stop the process
        # We do this by calling memory.set_complete()
        # WITHOUT THIS THE ROBOT WILL NEVER STOP!
        memory.set_complete()
        return memory

    # =========================================
    # 5. SET UP THE ROBOT AT INIT, TELLING IT EXACTLY WHAT TO USE
    # =========================================
    # WHEN BUILDING A ROBOT WE HAVE TO CALL SUPER() AND PASS IN THE MEMORY CLASS
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


# Nice clean version:


# =========================================
# HOW TO BUILD A ROBOT
# =========================================
# 1. INHERIT FROM AIRobot
class PoetryRobot(AIRobot):
    # ==========
    # 2. HERE'S WHAT YOU MUST BUILD FOR YOUR ROBOT
    # ==========
    ai_model: OpenAIChatCompletion  # <--- This is a subclass of BaseAIModel.  # You can make your own very easily or get one from robai.languagemodels
    memory: SimpleChatMemory  # <--- Start with SimpleChatMemory, # You can change this later
    input_model: ChatMessage  # <--- This is the input model for your robot. It's a ChatMessage by default, but you can change it to anything you like
    purpose: str = "I think you're a robot that writes poems from the user's input."
    pre_call_chain: List[Callable]  # <--- a list of functions
    post_call_chain: List[Callable]  # <--- a list of functions

    # =========================================
    # 3. What should your robot receive as input?
    # We're only defining this here to help us conceptualise what the robot will do.
    # SAY IT OUT LOUD - OUR ROBOT WILL RECEIVE A CHATMESSAGE AS INPUT!
    # =========================================
    class ChatMessage(BaseModel):
        role: str
        content: str

    # =========================================
    # 4. DEFINE THE PRE-CALL FUNCTIONS
    # =========================================

    # SIMPLE LOGS BEFORE THE ROBOT CALLS THE AI (if you want)
    def create_log(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        self.printer.pprint_color(
            f"{self.__class__.__name__} has just received some input, which is: {memory.input_model}".strip()
        )
        return memory

    # ADDING INTERACTION HISTORY TO MEMORY (if you want)
    def remember_user_interaction(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        memory.add_message_to_history(memory.input_model)
        return memory

    # CREATING INSTRUCTIONS OR 'PROMPT' FOR THE AI (you _must_ do this)
    def create_instructions(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        memory.instructions_for_ai: List[ChatMessage] = [
            memory.system_prompt,
            memory.input_model,
        ]
        self.ai_model.max_tokens = 30
        self.printer.pprint_color(
            f"{self.__class__.__name__} is about to call self.ai_model {self.ai_model} with instructions: {memory.instructions_for_ai}".strip()
        )
        return memory

    # =========================================
    # 5. DEFINE SOME POST-CALL FUNCTIONS
    # =========================================

    def print_what_I_did(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        self.printer.pprint_message(robot=self, message=memory.ai_response)
        return memory

    def stop_the_robot(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        # In our case, we don't want to do much else in post_call, so we just stop the process
        # We do this by calling memory.set_complete()
        # WITHOUT THIS THE ROBOT WILL NEVER STOP!
        memory.set_complete()
        return memory

    # =========================================
    # 5. SET UP THE ROBOT AT INIT, TELLING IT EXACTLY WHAT TO USE
    # =========================================
    # WHEN BUILDING A ROBOT WE HAVE TO CALL SUPER() AND PASS IN THE MEMORY CLASS
    def __init__(self):
        super().__init__(
            memory=SimpleChatMemory(
                purpose="You're a robot that only responds in poetry derived from the user's input."
            ),
            pre_call_chain=[
                self.create_log,
                self.remember_user_interaction,
                self.create_instructions,
            ],
            post_call_chain=[self.print_what_I_did, self.stop_the_robot],
            ai_model=OpenAIChatCompletion(),
        )


if __name__ == "__main__":
    poetry_robot = PoetryRobot()
    some_input_might_be = ChatMessage(
        role="user",
        content="I'm a user input. I heard that no matter what I say, you'll write a poem about it?",
    )
    result = poetry_robot.process(some_input_might_be)
