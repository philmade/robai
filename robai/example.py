from robai.schemas import ChatMessage, SystemMessage, AIMessage
from robai.chat import ChatRobot
from robai.protocols import ConsoleMessageHandler
from robai.func_tools import robot_function
from pydantic import BaseModel, Field
from typing import List
from pprint import pprint


# Simple Pydantic Models
class HelloWorldInput(BaseModel):
    message: str = Field(..., description="A message to print to the devs console")


class AddNumbersInput(BaseModel):
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")


class ArrayInput(BaseModel):
    items: List[str] = Field(..., description="A list of strings to process")


# Nested Pydantic Model Example
class SubItem(BaseModel):
    name: str = Field(..., description="Name of the sub-item")
    value: int = Field(..., description="Value of the sub-item")


class ComplexInput(BaseModel):
    title: str = Field(..., description="Title of the complex object")
    items: List[SubItem] = Field(..., description="List of sub-items to process")


# Add new Pydantic model for exit
class ExitInput(BaseModel):
    reason: str = Field(..., description="Reason for exiting the conversation")


class TestBot(ChatRobot[ChatMessage]):
    def __init__(self):
        super().__init__(message_handler=ConsoleMessageHandler())
        self.system_prompt = SystemMessage(
            content="""You are a helpful assistant testing our function framework. 
            If the user says 'bye', 'exit', 'quit' or similar, please use the exit_conversation function.
            When testing functions, please explain what you're doing and show the results."""
        )
        self.history_debug = True  # Add debug flag

    @robot_function()
    async def hello_world(self, input_data: HelloWorldInput) -> None:
        """Simple test function that prints a message"""
        print(input_data.message)
        self.function_results.append("hello_world", input_data.message)

    @robot_function()
    async def add_numbers(self, data: AddNumbersInput) -> None:
        """Test function showing parameter handling"""
        result = data.a + data.b
        self.function_results.append("add_numbers", f"Result: {result}")

    @robot_function()
    async def process_array(self, data: ArrayInput) -> None:
        """Test function showing array parameter handling"""
        result = ", ".join(data.items)
        self.function_results.append("process_array", f"Processed: {result}")

    @robot_function()
    async def process_complex(self, data: ComplexInput) -> None:
        """Test function showing nested object handling"""
        total = sum(item.value for item in data.items)
        result = f"Title: {data.title}, Total Value: {total}"
        items_summary = [f"{item.name}: {item.value}" for item in data.items]
        result += f"\nItems: {', '.join(items_summary)}"
        self.function_results.append("process_complex", result)

    @robot_function()
    async def chain_test(self, data: HelloWorldInput) -> None:
        """After this function, only hello_world will be available"""
        self.function_results.append(
            "chain_test", "Chain test complete. Now only hello_world is available."
        )

    @robot_function()
    async def single_use(self, data: HelloWorldInput) -> None:
        """This function will remove itself after use"""
        self.function_results.append(
            "single_use", "Function used and now removed from available functions."
        )

    @robot_function()
    async def exit_conversation(self, data: ExitInput) -> None:
        """Exit the conversation gracefully"""
        self.finished = True
        self.function_results.append(
            "exit_conversation", f"Conversation ended: {data.reason}"
        )

    async def stop_condition(self) -> bool:
        # Check for direct user exit commands
        if hasattr(self, "last_user_message"):
            user_input = self.last_user_message.content.lower()
            if user_input in ["bye", "exit", "quit"]:
                await self.exit_conversation(
                    ExitInput(reason="User requested exit directly")
                )
                return True
        return self.finished

    async def prepare(self) -> None:
        # Get the message
        message = await self.message_handler.wait_for_input()
        self.last_user_message = message

        # We handle history management here
        await self.message_handler.save_to_history(message)

        """Debug the history handling"""
        history = await self.message_handler.get_history(10)
        self.prompt_manager.set_system_prompt(self.system_prompt)
        for m in history:
            self.prompt_manager.add_message(m)

        if self.history_debug:
            # Print debug info about history
            print("\n=== History Debug ===")
            print(f"History length: {len(history)}")
            print(f"Last 3 messages: {history[-3:] if len(history) >= 3 else history}")
            print("===================\n")
            print("\n=== Prompt Manager Debug ===")
            print(f"Total messages: {len(self.prompt_manager.all_messages)}")
            pprint(f"All messages: {self.prompt_manager.all_messages}")
            print("==========================\n")

    async def process(self) -> None:
        # Handle initial response and any function calls
        await self._handle_streaming_response()

        # If we have pending function calls, process them and get AI response
        if self.pending_function_calls:
            # Update status with function results
            await self.message_handler.update_status(
                f"Function returned: {self.function_results.__str__()}"
            )

            # Add function results to context
            await self._add_function_results_to_context()

            # Create a new message to prompt AI to respond to the function results
            self.prompt_manager.add_message(
                SystemMessage(
                    content="Please respond to the function results above and tell the user what happened."
                )
            )

            # Get a new response from the AI about the function results
            await self.generate_ai_response()

            # Handle the new response, ensuring it's not empty
            response = await self._handle_streaming_response()

            if response and hasattr(response, "content"):
                # Save valid response to history
                await self.message_handler.save_to_history(response)
            else:
                # If we got an empty response, create a default one
                default_response = AIMessage(
                    role="assistant",
                    content="I've processed the function results. Would you like to try something else?",
                )
                await self.message_handler.save_to_history(default_response)

    async def finalize(self) -> None:
        pass
