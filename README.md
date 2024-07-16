
![Your RobAI Agent!](img/robai.png "Your RobAI Agent!")

# RobAI Framework

Welcome to RobAI - your friendly AI robot framework designed to make working with AI intuitive and easy! RobAI streamlines the construction and utilization of AI robots, offering a simple yet powerful paradigm to interact with AI models. Let's dive into the fun and engaging features that make RobAI the perfect companion for your AI development journey.

## Features

### 1. Before Call - AI Call - After Call Paradigm

RobAI introduces an elegant paradigm for AI interactions:

- **Before Call**: Set up your AI prompt and prepare any necessary data.
- **AI Call**: The AI processes the prompt and generates a response.
- **After Call**: Handle the AI's output and perform any post-processing steps.

### 2. Stop Condition

Easily define when the interaction with the AI should stop. This flexibility allows you to create AI workflows that can continue until a specific condition is met.

### 3. AI Description

With the `add_description` decorator, you can describe your AI functions in a way that makes them easily understandable and accessible. This feature enhances the readability and usability of your AI components.

## Getting Started

Here’s a simple example of a summarization robot that can summarize a large amount of text.

### Code Example: Summarization Robot

#### `summarization_robot.py`

```python
from robai.base import BaseRobot, BaseAI
from robai.schemas import ChatMessage
from typing import Any

class SummarizationRobot(BaseRobot):
    def load(self, input_data: str):
        self.input_data = input_data
        self.system_prompt = f"Please summarize the following text: {input_data}"

    def stop_condition(self) -> bool:
        return True  # Stop after one interaction for simplicity

    async def before_call(self, input_data: Any) -> None:
        user_message = ChatMessage(role="user", content=self.system_prompt)
        self.prompt.append(user_message)

    async def after_call(self, output_data: Any) -> None:
        print(f"Summary: {output_data}")

# Usage example
async def main():
    input_text = "Here is a long text that needs summarization..."
    summarization_robot = SummarizationRobot()
    summarization_robot.load(input_text)
    summary = await summarization_robot.interact()
    print(f"Summary: {summary}")

# Run the example
import asyncio
asyncio.run(main())
```

## Installation

To get started with RobAI, you can install it via pip:

```sh
pip install robai
```

## Contributing

We welcome contributions to RobAI! If you have ideas for new features, improvements, or bug fixes, please open an issue or submit a pull request on our GitHub repository.

## License

RobAI is open-source software licensed under the MIT License.

Embrace the future of AI development with RobAI. Let's build intelligent robots together! 🤖✨

For more information, visit our [GitHub repository](https://github.com/your-repo).

If you have any questions or need further assistance, feel free to reach out. Happy coding! 🚀





