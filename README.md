# Robai Framework

A powerful framework designed to streamline the construction and utilization of AI robots. By following the steps outlined below, developers can easily integrate AI functionalities into their applications and services.

## Installation

You can install Robai using pip, uv, or poetry:

### Using pip

```bash
pip install robai
```

### Using uv

```bash
uv pip install robai
```

### Using poetry

```bash
poetry add robai
```

## Quick Start

```python
from robai import BaseRobot, ChatMessage, SystemMessage

class MyRobot(BaseRobot):
    async def prepare(self):
        # Initialize your robot
        pass
        
    async def process(self):
        # Process user input
        pass
        
    def stop_condition(self):
        # Define when to stop
        return False

# Create and run your robot
robot = MyRobot()
await robot.interact()
```

## Requirements

- Python 3.11+
- OpenAI API key (set as environment variable `OPENAI_API_KEY`)

## License

MIT
