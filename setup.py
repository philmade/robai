from setuptools import setup, find_packages

setup(
    name="robai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=1.10.7",
        "faker>=19.3.0",
        "loguru>=0.7.0",
        "openai>=1.0.0",
        "tiktoken>=0.5.0",
        "fastapi>=0.100.0",
        "websockets>=11.0.0,<15.0.0",
        "rich>=13.0.0",
        "starlette",
    ],
    # Metadata
    author="philmade",
    description="RobAI is a powerful framework designed to streamline the construction and utilization of AI robots. By following the steps outlined below, developers can easily integrate AI functionalities into their applications and services.",
    keywords="robot ai framework",
    url="https://github.com/philmade/robai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
