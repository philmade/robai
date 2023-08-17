from ..memory import BaseMemory
from pydantic import BaseModel
from ..base import AIRobot
from ..in_out import ChatMessage, AIMessage
from ..languagemodels import (
    FakeAICompletion,
    OpenAIChatCompletion,
)  # Placeholder for an actual AI model
from typing import List
from faker import Faker

fake = Faker()

# =========================================
# CHAINED ROBOTS
# =========================================
# A robot is self cotained, so you're able to chain them together in the pre-call and post-call methods. Here is a simple example:

# ROBOT 1: Query Amplifier


class InputUserQuery(BaseModel):
    query_string: str
    age: int = None
    cookies: List[str] = None


class QueryAmplifierMemory(BaseMemory):
    input_model: InputUserQuery = None
    amplified_query: str = None
    instructions_for_ai: List[ChatMessage] = None


# PRE-CALL
def amplify_query(memory: QueryAmplifierMemory) -> QueryAmplifierMemory:
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
        ).dict()
    ]

    return memory


# POST-CALl
def add_response_to_memory(memory: QueryAmplifierMemory) -> QueryAmplifierMemory:
    # The Call() method always stores its results as a ChatMessage in memory.ai_response
    # We'll add that to out custom memory
    memory.amplified_query = memory.ai_response.content
    memory.set_complete()  # <--- Very important or we'll loop!
    return memory


# Construct The Robot
query_amplifier_robot = AIRobot(
    memory=QueryAmplifierMemory(
        purpose="Your job is to amplify the user's query with ideas, creativity, context and even some specifics"
    ),
    pre_call_chain=[amplify_query],
    post_call_chain=[add_response_to_memory],
    ai_model=OpenAIChatCompletion,  # Placeholder for an actual AI model
)

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


# PRE-CALL
def get_amplified_query(memory: LibraryGuyMemory) -> LibraryGuyMemory:
    # Here we are calling our first robot to amplify the user's query
    query_amplifier_robot.ai_model.max_tokens = 100
    amplified_memory: QueryAmplifierMemory = query_amplifier_robot.process(
        memory.input_model
    )
    memory.instructions_for_ai = [
        ChatMessage(
            role="user",
            content=f"""
    Craft a perfect SQL query for this user query: {amplified_memory.amplified_query} 
    Consider the user's context:
    User is {memory.input_model.age or 'Not Available'} years old
    User has cookies: {memory.input_model.cookies or 'Not Available'}
    Reply ONLY in the form of a SQL query.
    """,
        ).dict()
    ]
    return memory


# POSTCALL
def add_response_to_memory(memory: LibraryGuyMemory) -> LibraryGuyMemory:
    memory.sql_query = memory.ai_response.content
    return memory


def fetch_books_from_library(memory: LibraryGuyMemory) -> LibraryGuyMemory:
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


# Re-construct The Library Guy Robot
library_guy_robot = AIRobot(
    memory=LibraryGuyMemory(purpose="Library Book Recommendations"),
    pre_call_chain=[get_amplified_query],
    post_call_chain=[add_response_to_memory, fetch_books_from_library],
    ai_model=OpenAIChatCompletion,  # Placeholder for an actual AI model
)

# Re-test The Library Guy Robot
test_library_query = InputUserQuery(
    query_string="great books", age=25, cookies=["young_adult", "fiction_lover"]
)
memory_with_library_recommendations = library_guy_robot.process(test_library_query)

# pprint(memory_with_library_recommendations.dict())
# query = InputUserQuery(query_string="great books", age=25, cookies=["young_adult", "fiction_lover"])
# query_amplifier_robot.ai_model.max_tokens = 100
# amplified_memory = query_amplifier_robot.process(query)
