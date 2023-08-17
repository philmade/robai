# Now let's re-import the necessary modules and classes, and then re-run the Weather Information Robot example.
from ..memory import BaseMemory
from pydantic import BaseModel
from ..base import AIRobot
from ..languagemodels import FakeAICompletion  # Placeholder for an actual AI model
from typing import List
from pprint import pprint


# Define the input model for the Recipe Suggestion Robot
class RecipeQuery(BaseModel):
    ingredients: List[str]
    cuisine: str


# Define the memory for the Recipe Suggestion Robot
class RecipeMemory(BaseMemory):
    input_model: RecipeQuery = None
    recipe_suggestion: str = None
    detailed_recipe: str = None


# Fake database to simulate AI's response
class RecipeDB:
    @staticmethod
    def get_recipe_suggestion(ingredients: List[str], cuisine: str) -> str:
        # Simple simulation of an AI's response
        main_ingredient = ingredients[0]
        if cuisine == "Italian":
            return f"How about a {main_ingredient} pasta?"
        elif cuisine == "Chinese":
            return f"How about a {main_ingredient} stir fry?"
        else:
            return f"How about a {main_ingredient} salad?"

    @staticmethod
    def get_detailed_recipe(suggestion: str) -> str:
        # Another simple simulation of an AI's response
        return f"Recipe for {suggestion}: ... [detailed steps here] ..."


# Pre-call chain functions
def extract_user_query(memory: RecipeMemory) -> RecipeMemory:
    memory.instructions_for_ai = f"Find recipes with {', '.join(memory.input_model.ingredients)} for {memory.input_model.cuisine} cuisine."
    return memory


def get_recipe_suggestion(memory: RecipeMemory) -> RecipeMemory:
    memory.recipe_suggestion = RecipeDB.get_recipe_suggestion(
        memory.input_model.ingredients, memory.input_model.cuisine
    )
    memory.instructions_for_ai = f"Get detailed recipe for {memory.recipe_suggestion}"
    return memory


# Post-call chain function
def get_detailed_recipe(memory: RecipeMemory) -> RecipeMemory:
    memory.detailed_recipe = RecipeDB.get_detailed_recipe(memory.recipe_suggestion)
    memory.set_complete()
    return memory


# Construct the Recipe Suggestion Robot
recipe_robot = AIRobot(
    memory=RecipeMemory(purpose="Recipe Suggestion"),
    pre_call_chain=[extract_user_query, get_recipe_suggestion],
    post_call_chain=[get_detailed_recipe],
    ai_model=FakeAICompletion,
)

# Test the Recipe Suggestion Robot
test_recipe_query = RecipeQuery(
    ingredients=["chicken", "tomatoes", "onions"], cuisine="Italian"
)
memory_with_recipe = recipe_robot.process(test_recipe_query)

pprint(memory_with_recipe.dict())
