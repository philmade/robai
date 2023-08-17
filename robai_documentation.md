# RobAI Documentation

## in_out.py

## memory.py

## languagemodels.py

## chains.py

## full_example.py

## setup.py

## base.py

## errors.py


### Nuances for in_out.py:

1. **Default Values**: Both `ChatMessage` and `AIMessage` have default values set for their fields. Users should be aware of these defaults, especially when constructing new instances without providing specific values.
2. **Message Structure**: The `history` field in the `Conversation` class expects a list of `ChatMessage` objects. When interfacing with an AI model, these objects might need to be converted to dictionaries using the `.dict()` method.
    
### Nuances for memory.py:

1. **Purpose Field**: The `purpose` field in the `BaseMemory` class serves as a prompt for the system. Users should provide a value that clearly describes the robot's purpose.
2. **Input Model**: The default input model is `ChatMessage`. Users might want to provide custom models based on specific requirements and ensure compatibility with the rest of the system.
3. **Instructions for AI**: The `instructions_for_ai` field expects a list of `ChatMessage` objects. When interfacing with certain AI models, these objects might need to be converted to dictionaries using the `.dict()` method.
    
### Nuances for languagemodels.py:

1. **Abstract Base Class**: The `BaseAIModel` is an abstract class, meaning you cannot instantiate it directly. It's intended as a base for other classes implementing actual AI models.
2. **Instructions for AI Type**: The class variable in `BaseAIModel` hints at a type specification for instructions given to AI models.
    
### Nuances for chains.py:

1. **Input Model Assumption**: The `create_instructions` function assumes the `input_model` attribute of the `memory` object (of type `BaseMemory`) is of type `ChatMessage`.
2. **Mutating BaseMemory**: The function modifies the provided `BaseMemory` object, adding a message and setting the instructions for the AI.
3. **Message as Instruction**: This function sets the provided message as the instruction for the AI.
    
### Nuances for full_example.py:

1. **Data Model**: The `ClothesBasket` class represents the data structure. The `clothing_items` and `clothing_costs` lists should be of equal length, as each item corresponds to a cost.
2. **Custom Model**: This file demonstrates creating a custom data model using Pydantic's `BaseModel`.
3. **Comments as Guide**: The file is structured with comments guiding through different steps of the implementation.
    
### Nuances for setup.py:

1. **Dependencies**: The package relies on "pydantic". During installation, "pydantic" will also be installed.
2. **Packaging and Distribution**: Users can package and distribute the "robai" package using this setup file.
    
### Nuances for base.py:

1. **Dependencies**: The `AIRobot` class is dependent on other modules like `languagemodels`, `errors`, and `memory`.
2. **Initialization**: The `AIRobot` class requires specific parameters for initialization.
3. **Pre and Post Call Chains**: Users can provide chains of functions to be executed before and after the AI call.
    
### Nuances for errors.py:

1. **Serialization Requirement**: When setting up chains for preprocessing data before the AI call, ensure the final function in the chain returns a JSON-serializable object.
2. **Initialization Importance**: Proper initialization of the `AIRobot` is essential.
    