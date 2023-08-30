from robai.memory import BaseMemory
from pydantic import BaseModel
from robai.base import AIRobot
from robai.in_out import ChatMessage
from robai.languagemodels import (
    OpenAIChatCompletion,
)  # Placeholder for an actual AI model
from typing import List, Callable
from faker import Faker
from robai.utility import pprint_color

fake = Faker()


# =========================================
# CHAINED ROBOTS
# =========================================
# Once constructed, a robot is self contained, so you're able to chain them together like ordinary functions
# in the pre-call and post-call methods. Here is a simple example:

# ROBOT 1: Query Amplifier


# All robots must have an input model - by default it's a ChatMessage, but we will make a custom one:
class InputUserQuery(BaseModel):
    query_string: str
    age: int = None
    cookies: List[str] = None


# Memory
class QueryAmplifierMemory(BaseMemory):
    input_model: InputUserQuery = None
    amplified_query: str = None
    instructions_for_ai: List[ChatMessage] = None


# Construct The Robot
class QueryAmplifierRobot(AIRobot):
    memory: QueryAmplifierMemory
    pre_call_chain: List[Callable]
    post_call_chain: List[Callable]
    ai_model = OpenAIChatCompletion  # Placeholder for an actual AI model

    # PRE-CALL
    def amplify_query(self, memory: QueryAmplifierMemory) -> QueryAmplifierMemory:
        memory.instructions_for_ai = [
            ChatMessage(
                role="user",
                content=f"""
        A user is searching for: '{memory.input_model.query_string}'
        User is {memory.input_model.age or 'Not Available'} years old
        User has cookies: {memory.input_model.cookies or 'Not Available'}
        Creatively augment their query so with suggestions you think will resonate with them. 
        If it's books, suggest genres, topics, authors, etc. If its films, do the same. 
        You get the idea, whatever the query is, amplify and augment it creatively so their query returns great suggestions.
        Your suggestions will be sent to another AI model that will craft a perfect SQL query from your suggestions.
        Your objective is to be creative and reply with a dense set of suggestions.
        """,
            )
        ]

        return memory

    # POST-CALl
    def add_response_to_memory(
        self, memory: QueryAmplifierMemory
    ) -> QueryAmplifierMemory:
        # The Call() method always stores its results as a ChatMessage in memory.ai_response
        # We'll add that to out custom memory
        memory.amplified_query = memory.ai_response.content
        memory.set_complete()  # <--- Very important or we'll loop!
        self.pprint_message(message=memory.ai_response)
        return memory

    def __init__(self):
        super().__init__(
            memory=QueryAmplifierMemory(
                purpose="Your job is to amplify the user's query with ideas, creativity, context and even some specifics"
            ),
            pre_call_chain=[self.amplify_query],
            post_call_chain=[self.add_response_to_memory],
            ai_model=OpenAIChatCompletion(),  # Real OPENAI model
        )


query_amplifier_robot = QueryAmplifierRobot()
# memory = QueryAmplifierMemory()
# ROBOT 2: The Library Guy


# Define its memory.
class LibraryGuyMemory(BaseMemory):
    input_model: InputUserQuery = (
        None  # <--- We'll re-use the same input model as the Query Amplifier Robot
    )
    sql_query: str = None
    recommendations: List[str] = []
    library_query_result: str = None
    instructions_for_ai: List[ChatMessage] = None


class LibraryGuyRobot(AIRobot):
    memory: LibraryGuyMemory = LibraryGuyMemory(purpose="Library Book Recommendations")
    ai_model = OpenAIChatCompletion  # Placeholder for an actual AI model

    # PRE-CALL
    def get_amplified_query(self, memory: LibraryGuyMemory) -> LibraryGuyMemory:
        # Here we are calling our first robot to amplify the user's query
        query_amplifier_robot.ai_model.max_tokens = 100
        # amplified_memory = query_amplifier_robot.process(
        #     memory.input_model
        # )
        amplifier_memory = query_amplifier_robot.process(memory.input_model)
        memory.instructions_for_ai = [
            ChatMessage(
                role="user",
                content=f"""
        Craft a perfect SQL query for this user query: {amplifier_memory.ai_response.content} 
        Consider the user's context:
        User is {memory.input_model.age or 'Not Available'} years old
        User has cookies: {memory.input_model.cookies or 'Not Available'}
        Reply ONLY in the form of a SQL query.
        """,
            )
        ]
        return memory

    # POSTCALL
    def add_response_to_memory(self, memory: LibraryGuyMemory) -> LibraryGuyMemory:
        memory.sql_query = memory.ai_response.content
        return memory

    def fetch_books_from_library(self, memory: LibraryGuyMemory) -> LibraryGuyMemory:
        # Fake database to simulate Library esponse
        class LibraryDB:
            @staticmethod
            def run_query(query: str) -> str:
                fake_response = " ".join(fake.sentences(nb=5))
                return fake_response

        library_db = LibraryDB()
        memory.library_query_result = library_db.run_query(memory.sql_query)
        memory.set_complete()
        return memory

    def __init__(self):
        super().__init__(
            memory=LibraryGuyMemory(purpose="Library Book Recommendations"),
            pre_call_chain=[self.get_amplified_query],
            post_call_chain=[
                self.add_response_to_memory,
                self.fetch_books_from_library,
            ],
            ai_model=OpenAIChatCompletion(),  # Placeholder for an actual AI model
        )


library_guy_robot = LibraryGuyRobot()

if __name__ == "__main__":
    test_library_query = InputUserQuery(
        query_string="great books", age=25, cookies=["young_adult", "fiction_lover"]
    )
    memory_with_library_recommendations = library_guy_robot.process(test_library_query)

    library_guy_robot.pprint_message(message=library_guy_robot.memory.ai_response)
