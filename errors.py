# 1. Define a custom exception
class SerializationError(Exception):
    def __init__(
        self,
        message="The output of the pre_call_chain is not JSON-serializable. Ensure the final function in the chain returns a serializable object.",
    ):
        self.message = message
        super().__init__(self.message)


class AIRobotInitializationError(Exception):
    def __init__(
        self,
        message="The AIRobot was not initialized correctly. Please ensure that the AIRobot is initialized with the correct parameters.",
    ):
        self.message = message
        super().__init__(self.message)
