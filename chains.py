from robai.memory import BaseMemory


# EMPTY CHAIN
async def empty_pre_call_chain(memory: BaseMemory):
    memory.set_complete()
    return input, memory


do_nothing = [empty_pre_call_chain]
