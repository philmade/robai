from robai.languagemodels import BaseAIModel, OpenAIChatCompletion
from robai.errors import AIRobotInitializationError, SerializationError
from robai.memory import BaseMemory, ChatMessage, SimpleChatMemory
from robai.chains import do_nothing_in_post_call, simply_create_instructions
from robai.utility import CustomConsole
from typing import Any, List, Callable, Generator, Self
from pydantic import BaseModel
import inspect
from loguru import logger
import asyncio
import json
from abc import ABC
import random
from rich.console import Console
from rich.theme import Theme
from rich.pretty import pprint


class AIRobot(ABC):
    def __init__(
        self,
        memory: BaseMemory = SimpleChatMemory(),
        ai_model: BaseAIModel = OpenAIChatCompletion(),
        pre_call_chain: List[Callable] = simply_create_instructions,
        post_call_chain: List[Callable] = do_nothing_in_post_call,
        robots_functions: List[Callable] = None,
        logging_enabled: bool = True,
        **kwargs,
    ) -> Self:
        self.pre_call_chain = pre_call_chain
        self.post_call_chain = post_call_chain
        self.robots_functions = robots_functions or []
        self.memory: BaseMemory = memory
        if self.memory.purpose is None or "":
            raise AIRobotInitializationError(
                message="Memory must have a purpose to be used by a robot"
            )
        self.ai_model: BaseAIModel = ai_model
        # self.printer = MessagePrinter()
        self.theme = Theme(
            {
                "red": "red",
                "blue": "blue",
                "green": "green",
                "yellow": "yellow",
                "cyan": "cyan",
                "magenta": "magenta",
            }
        )
        # SET THE SYSTEM PROMPT - the purpose of the robot in a message format.
        system_prompt = ChatMessage(
            role="system",
            content=f"You are {self.__class__.__name__}, your purpose is: {self.memory.purpose}",
        )
        self.memory.add_message_to_history(system_prompt)
        self.memory.system_prompt = system_prompt
        self.color = random.choice(
            ["red", "blue", "green", "yellow", "cyan", "magenta"]
        )
        self.logging_enabled = logging_enabled
        self.console = CustomConsole(theme=self.theme, width=100)
        if self.logging_enabled:
            self.console.rule(f"[cyan]Initializing {self.__class__.__name__}")
            self.console.pprint(self.memory.purpose)
            self.console.pprint(self.memory)
            self.console.rule("[cyan]Initialization Complete")

    def process(self, input: BaseModel, stream: bool = False) -> BaseMemory:
        """
        # Input should be the same as self.memory.input_model
        ## returns an instance of self.memory
        """
        if self.logging_enabled:
            self.console.rule("[green]Starting Pre-call Chain")
        expected_type = self.memory.__annotations__["input_model"]
        if not isinstance(input, expected_type):
            raise TypeError(
                f"Expected input of type {expected_type}, but got {type(input)}"
            )
        # IMPORTANT STEP - THE INPUT MUST BE ADDED TO THE MEMORY
        self.memory.input_model = input
        # THE USER CAN DO WHATEVER THEY WANT FROM IT FROM THERE
        self.memory.complete = False
        while not self.memory.complete:
            # Pre-call chain
            for function in self.pre_call_chain:
                try:
                    if inspect.isgeneratorfunction(function):  # Generator function
                        self.memory.set_complete()
                        gen = function(self.memory)
                        input_model_processed = [
                            item for item in gen
                        ]  # Consume generator

                    else:  # Regular synchronous function
                        if self.logging_enabled:
                            self.console.print(f"calling [yellow]{function.__name__}()")
                        self.memory = function(self.memory)
                except AttributeError as e:
                    logger.info(
                        f"Attribute error in {function.__name__} in pre-call chain: {e}. Did you forget to return memory!?"
                    )
                except Exception as e:
                    logger.info(
                        f"Exception in {function.__name__} in pre-call chain: {e}"
                    )
            if self.logging_enabled:
                self.console.rule("[white]DONE WITH PRE-CALL")

            # Call to AI model
            if self.logging_enabled:
                self.console.rule("[green]AI Model Call")
                with self.console.status("[blue]Calling AI Model..."):
                    self.memory = self.ai_model.call(self.memory)
                    if stream:
                        streamed_response: Generator = self.ai_model.call_manager(
                            self.memory
                        )
                    else:
                        self.memory = self.ai_model.call_manager(memory=self.memory)
            else:
                if stream:
                    streamed_response: Generator = self.ai_model.call_manager(
                        self.memory
                    )
                else:
                    self.memory = self.ai_model.call_manager(memory=self.memory)

            if self.logging_enabled:
                self.console.print("[white]CALLED AI MODEL")

            # Post-call chain
            if self.logging_enabled:
                self.console.rule("[blue]Starting Post-call Chain")

            for function in self.post_call_chain:
                try:
                    if inspect.isgeneratorfunction(function):  # Generator function
                        self.memory.set_complete()
                        gen = function(streamed_response, self.memory)
                        return gen
                    else:  # Regular synchronous function
                        if self.logging_enabled:
                            self.console.print(f"calling [yellow]{function.__name__}()")
                            self.memory = function(self.memory)

                except AttributeError as e:
                    logger.info(
                        f"Attribute error in {function.__name__} in post-call chain: {e}. Did you forget to return memory!?"
                    )
                except Exception as e:
                    logger.info(
                        f"Exception in {function.__name__} in post-call chain: {e}"
                    )
            if self.logging_enabled:
                self.console.rule("[white]DONE WITH POST-CALL")

            if self.logging_enabled:
                self.console.rule("[magenta]Processing Complete - returning memory")

            return self.memory

    async def aprocess(self, input_data: Any, stream: bool = False) -> BaseModel:
        if not isinstance(input_data, type(self.memory.input_model)):
            raise TypeError(
                f"Robot recieved {type(input_data)} as input, but it expected what's in memory.input_model which is type {type(self.memory.input_model)}"
            )
        # IMPORTANT STEP - THE INPUT MUST BE ADDED TO THE MEMORY
        self.memory.input_model = input_data
        # THE USER CAN DO WHATEVER THEY WANT FROM IT FROM THERE
        self.memory.complete = False
        while not self.memory.complete:
            # Pre-call chain
            for function in self.pre_call_chain:
                try:
                    if asyncio.iscoroutinefunction(function):  # Async function
                        self.memory = await function(self.memory)
                    elif inspect.isgeneratorfunction(function):  # Generator function
                        self.memory.set_complete()
                        gen = function(self.memory)
                        input_model_processed = [
                            item for item in gen
                        ]  # Consume generator

                    else:  # Regular synchronous function
                        self.memory = function(self.memory)
                except Exception as e:
                    logger.info(
                        f"Exception in {function.__name__} in pre-call chain: {e}"
                    )

            try:
                json.dumps(self.memory.instructions_for_ai)
            except TypeError:
                raise SerializationError(
                    message=f"Your final function in pre-call isn't setting memory.instructions_for_ai to something that's JSON serializable, got: {input_model_processed} instead"
                )

            # Call to AI model
            if stream:
                streamed_response: Generator = self.ai_model.stream_call(self.memory)
            else:
                self.memory: BaseMemory = self.ai_model.call(self.memory)

            # Post-call chain
            for function in self.post_call_chain:
                try:
                    if asyncio.iscoroutinefunction(function):  # Async function
                        self.memory = await function(self.memory)
                    elif inspect.isgeneratorfunction(function):  # Generator function
                        self.memory.set_complete()
                        gen = function(streamed_response, self.memory)
                        return gen
                    else:  # Regular synchronous function
                        self.memory = function(self.memory)
                except Exception as e:
                    logger.info(
                        f"Exception in {function.__name__} in post-call chain: {e}"
                    )

        return self.memory

    def message_history_strings(self) -> str:
        return "\n".join(
            [message.content for message in self.memory.message_history if message]
        )

    def robot_call_header(self, robot: "AIRobot") -> str:
        return f"\n\nROBOT API CALL INSTRUCTIONS: YOU ARE IN A ROBOT TO ROBOT API CALL. WHEN THE EXCHANGE IS DONE, YOU MUST SAY THE WORD ROBOCALL-END TO END THE CALL."

    def check_end_call(self, text: str, keyword="ROBOCALL-END"):
        """
        Check if the keyword appears twice in the text.

        Args:
        - text (str): The text to search in.
        - keyword (str): The keyword to search for. Default is "DONE".

        Returns:
        - bool: True if the keyword appears twice, False otherwise.
        """
        # Strip the message of the protocol
        stripped_text = text.split("ROBOT API CALL INSTRUCTIONS")[0]

        return keyword in stripped_text

    def robot_call_first_message(self, robot_i_am_calling: "AIRobot") -> ChatMessage:
        content = (
            "Hello, "
            + robot_i_am_calling.__class__.__name__
            + " I am "
            + self.__class__.__name__
            + "."
            + "\nCan you help me with my task?"
            + "\nMy task was to is: "
            + self.memory.purpose
            + "\nI received this request: "
            + str(self.memory.input_model.dict())
            + "\nThis was my response: "
            + self.memory.ai_response.content
            + "\nCan you help me with it?"
            + "\n"
            + self.robot_call_header(robot_i_am_calling)
        )

        return ChatMessage(role="system", content=content)

    def robo_call(
        self,
        robot_I_want_to_call: "AIRobot",
    ) -> None:
        """
        Handles the robot-to-robot communication. The calling robot sends itself as the argument,
        allowing the current robot to access its memory and responses.
        """
        max_iterations = 10
        iteration_count = 0
        self.memory.in_robo_call = True
        robot_I_want_to_call.memory.in_robo_call = True
        while self.memory.in_robo_call and iteration_count < max_iterations:
            iteration_count += 1
            # pprint_color("Iteraction count: " + str(iteration_count))
            if (
                self.robot_call_header(robot_I_want_to_call)
                not in self.message_history_strings()
            ):
                # IF WE HAVEN'T STARTED THE CALL YET WE START IN THIS BLOCK
                # First we'll add the message we intend to send to our history:
                robot_call_first_message = self.robot_call_first_message(
                    robot_i_am_calling=robot_I_want_to_call
                )
                # Now we will actually send that message to the robot
                # We do that by sending a special message and storing the response
                # at self.memory.robot_response
                self.memory.robot_response = robot_I_want_to_call.robo_recieve(
                    robot_who_is_calling=self,
                    message=robot_call_first_message,
                )
                self.memory.add_message_to_history(robot_call_first_message)
                self.memory.add_message_to_history(self.memory.robot_response)
                self.pprint_message(robot=self, message=robot_call_first_message)
                self.pprint_message(
                    robot=robot_I_want_to_call, message=self.memory.robot_response
                )

            else:
                # ELSE WE ARE ALREADY IN THE CALL
                # HENCE, WE NEED TO PROCESS THE MESSAGE AND RETURN A RESPONSE
                # STEP ONE - DID THE ROBOT I CALLED HANG UP?
                if self.memory.robot_response:
                    if self.check_end_call(self.memory.robot_response.content):
                        self.pprint_message(
                            robot=self, message=self.memory.robot_response
                        )
                        self.memory.in_robo_call = False
                        self.memory.set_complete()
                        return self.memory, ChatMessage(
                            role="system",
                            content="ROBOCALL-END",
                        )
                        break
                # STEP TWO - IF NOT, WE NEED TO CALL OUR AI MODEL WITH THE ROBO-CALL CHAT HISTORY AS CONTEXT
                self.memory.instructions_for_ai = self.memory.message_history[-5:]
                # That's a good standard context for a robot exchange
                # Now we call the AI model with that context
                self.memory = self.ai_model.call(self.memory)
                my_reply = self.memory.ai_response
                my_reply.role = "system"
                # SOMETIMES THE AI PUTS THE HEADER IN ITSELF
                if self.robot_call_header(robot_I_want_to_call) not in my_reply.content:
                    my_reply.content += "\n\n" + self.robot_call_header(
                        robot_I_want_to_call
                    )
                self.memory.add_message_to_history(my_reply)
                self.memory.add_message_to_history(self.memory.robot_response)
                self.pprint_message(robot=self, message=my_reply)
                self.memory.robot_response = robot_I_want_to_call.robo_recieve(
                    robot_who_is_calling=self, message=my_reply
                )
                self.pprint_message(
                    robot=robot_I_want_to_call,
                    message=self.memory.robot_response,
                )

    def robo_recieve(
        self, robot_who_is_calling: "AIRobot", message: ChatMessage
    ) -> ChatMessage:
        while self.memory.in_robo_call:
            # HAS THE CALLER ALREADY HUNG UP?
            if message:
                if self.check_end_call(message.content):
                    self.memory.in_robo_call = False
                    self.memory.set_complete()
                    return ChatMessage(
                        role="system",
                        content="ROBOCALL-END",
                    )
                    break
            # IF NOT - LETS PROCESS THEIR REQUEST
            # THE SYSTEM PROMPT AND THE ROBOT'S INCOMING MESSAGE ARE THE CONTEXT FOR THE AI CALL
            self.memory.add_message_to_history(message)
            self.memory.instructions_for_ai = self.memory.message_history[-5:]
            self.memory = self.ai_model.call(self.memory)
            message_to_send_back = self.memory.ai_response
            message_to_send_back.role = "system"
            if (
                self.robot_call_header(robot_who_is_calling)
                not in message_to_send_back.content
            ):
                message_to_send_back.content += self.robot_call_header(
                    robot_who_is_calling
                )
            self.memory.add_message_to_history(message_to_send_back)
            return message_to_send_back
