FROM python:3.11-slim

# Install Poetry
RUN pip install poetry

# Set working directory
WORKDIR /app

# Copy only the files needed for installing dependencies
COPY pyproject.toml poetry.lock README.md ./

# Configure Poetry to not create a virtual environment inside the container
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev --no-interaction

# Copy the rest of the application
COPY . .

# Expose port if needed (adjust as necessary)
EXPOSE 8000

# Command to run the application (adjust as necessary)
CMD ["poetry", "run", "python", "-m", "robai"] 