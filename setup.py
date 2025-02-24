from setuptools import setup, find_packages

setup(
    name="robai",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here
        "pydantic",
        # ...
    ],
    # Metadata
    author="philmade",
    description="RobAI is a powerful framework designed to streamline the construction and utilization of AI robots. By following the steps outlined below, developers can easily integrate AI functionalities into their applications and services.",
    keywords="robot ai framework",
    url="https://github.com/philmade/robai",
    classifiers=[
        # List of classifiers: https://pypi.org/classifiers/
    ],
)
