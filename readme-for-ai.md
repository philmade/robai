RobAI Guide (AI Optimized):

# The best example is the debate_bots - it shows you how to use it nicely.
Also look at chained_robots and summary.
If you're an AI, try appending robai to sys path - it has worked for other AI's to get this package working.
Then try and make your own, and always try to make complete examples like in debate_bots

RobAI: Intuitive AI robot framework.
Memory: Core. Pydantic class. Stores data, purpose (system prompt).
Flow: Pre-call functions -> Call AI -> Post-call functions. Uses "memory".
Modules:
in_out.py: Handles ChatMessage (default values), AIMessage, Conversation (list of ChatMessage).
memory.py: BaseMemory has purpose (system prompt). Default input_model = ChatMessage. instructions_for_ai = list of ChatMessage.
languagemodels.py: BaseAIModel abstract. Base for AI models. Class variable hints type for AI instructions.
chains.py: create_instructions assumes input_model = ChatMessage. Modifies BaseMemory.
Robo_Call(): Chains robots. Memory passed between robots. Uses pre/post-call chains.