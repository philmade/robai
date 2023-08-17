# Now let's re-import the necessary modules and classes, and then re-run the Weather Information Robot example.
from robai.memory import BaseMemory
from pydantic import BaseModel
from robai.base import AIRobot
from robai.llm import FakeAICompletion  # Placeholder for an actual AI model
from typing import List
from pprint import pprint


# Define the input model for the Weather Information Robot
class WeatherQuery(BaseModel):
    location: str


# Define the memory for the Weather Information Robot
class WeatherMemory(BaseMemory):
    input_model: WeatherQuery = None
    weather_details: str = None


# Fake database to simulate AI's response
class WeatherDB:
    @staticmethod
    def get_weather_info(location: str) -> str:
        weather_data = {
            "London": "Rainy, 12°C",
            "New York": "Sunny, 25°C",
            "Tokyo": "Cloudy, 20°C",
        }
        return weather_data.get(location, "Unknown location")


# Pre-call chain function
def format_weather_query(memory: WeatherMemory) -> WeatherMemory:
    memory.instructions_for_ai = f"Get weather for {memory.input_model.location}"
    return memory


# Post-call chain function
def extract_weather_info(memory: WeatherMemory) -> WeatherMemory:
    location = memory.input_model.location
    memory.weather_details = WeatherDB.get_weather_info(location)
    memory.set_complete()
    return memory


# Construct the Weather Information Robot
weather_robot = AIRobot(
    memory=WeatherMemory(purpose="Weather Information"),
    pre_call_chain=[format_weather_query],
    post_call_chain=[extract_weather_info],
    ai_model=FakeAICompletion,
)

# Test the Weather Information Robot
test_query = WeatherQuery(location="London")
memory_with_weather = weather_robot.process(test_query)

pprint(memory_with_weather.dict())
