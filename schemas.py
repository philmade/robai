from pydantic import BaseModel
import tiktoken
from typing import List, Union, Dict

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


class TokenBaseModel(BaseModel):
    def token_count(self) -> int:
        return len(tokenizer.encode(self.model_dump_json()))


class ChatMessage(TokenBaseModel):
    role: str = "user"
    robot: str = "case.bot"
    content: str = "Hello, I am a user"


class AIMessage(TokenBaseModel):
    role: str = "ai"
    robot: str = "case.bot"
    content: str = "Hello, I am an AI"


class SystemMessage(TokenBaseModel):
    role: str = "system"
    robot: str = "case.bot"
    content: str = "Hello, I am a system message"


class FunctionResults(TokenBaseModel):
    role: str = "system"
    robot: str = "case.bot"
    content: str = "Use me to store function results"


class ToolCallResults(BaseModel):
    task: Union[Dict[str, str], str] = {}
    task_history: List[str] = []
    log: list[str] = []
    errors: list[str] = []
    current_case_title: str = (
        "NO CASE TITLE SET - do not add sources until a case title is set"
    )
    current_case_id: Union[str, int] = (
        "NO CASE ID SET - do not add sources until a case ID is set"
    )
    notes: List[str] = []
    function_results: dict = {}

    def summarize(self):
        """
        Summarize the details of the tool call results.
        """
        # Shorten the notes and function results for summarization
        self.notes = [
            note[:50] + "..." if len(note) > 50 else note for note in self.notes
        ]
        self.function_results = {
            key: (str(value)[:50] + "..." if len(str(value)) > 50 else value)
            for key, value in self.function_results.items()
        }
        # Summarize the task and task history
        summarized_tasks = {
            key: (value[:50] + "..." if len(value) > 50 else value)
            for key, value in self.task.items()
        }
        self.task = summarized_tasks
        self.task_history = [
            task[:50] + "..." if len(task) > 50 else task for task in self.task_history
        ]
