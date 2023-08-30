from typing import List, Dict, Callable

# 1. Robot Registry


class RobotRegistry:
    robots: Dict[str, List[str]] = {}  # Robot name -> List of capabilities

    @classmethod
    def register_robot(cls, name: str, capabilities: List[str]):
        cls.robots[name] = capabilities

    @classmethod
    def search_capability(cls, capability: str) -> List[str]:
        return [name for name, caps in cls.robots.items() if capability in caps]


# 2. Basic Robots


class MathRobot:
    name = "MathRobot"
    capabilities = ["addition", "subtraction"]

    @staticmethod
    def add(x, y):
        return x + y

    @staticmethod
    def subtract(x, y):
        return x - y


class TextRobot:
    name = "TextRobot"
    capabilities = ["reverse"]

    @staticmethod
    def reverse(text: str) -> str:
        return text[::-1]


# Registering the robots
RobotRegistry.register_robot(MathRobot.name, MathRobot.capabilities)
RobotRegistry.register_robot(TextRobot.name, TextRobot.capabilities)


# 3. Complex Robot


class ComplexRobot:
    @staticmethod
    def complex_task(x, y, text):
        # Try to add numbers
        robots_for_addition = RobotRegistry.search_capability("addition")
        if robots_for_addition:
            result_add = getattr(globals()[robots_for_addition[0]], "add")(x, y)
        else:
            result_add = "No robot capable of addition found."

        # Try to reverse text
        robots_for_reverse = RobotRegistry.search_capability("reverse")
        if robots_for_reverse:
            result_text = getattr(globals()[robots_for_reverse[0]], "reverse")(text)
        else:
            result_text = "No robot capable of reversing text found."

        return result_add, result_text


# Testing the ComplexRobot
result = ComplexRobot.complex_task(5, 3, "hello")
result
