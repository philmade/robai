from typing import List
from robai.memory import BaseMemory
from robai.in_out import ChatMessage


class SimpleChatMemory(BaseMemory):
    input_model: List[ChatMessage] = None
    instructions_for_ai: List[ChatMessage] = None


class SummaryRobotMemory(BaseMemory):
    input_model: str = None
    chunks: List[str] = []
    # The context window for gpt3.5 is now 16,000 tokens. Each chunk must be less than that.
    # Assumes the model will summarise 10,000 tokens to 6,000 tokens.
    chunk_length_limit: int = 10000
    summaries: List[str] = []
    current_chunk_index: int = 0
    total_chunks: int = 0
    instructions_for_ai: List[ChatMessage] = None
