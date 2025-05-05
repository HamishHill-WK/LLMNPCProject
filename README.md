# LLM NPC - Dynamic NPC prototype

## Setup and Usage

### Requirements

- Python 3.8+
- Flask and related web packages
- Access to Ollama for local LLM hosting or OpenAI API key
- Download Ollama here https://ollama.com/
- once installed run

        ollama run deepseek-r1:8b


### Running the Application

1. Ensure all required components are installed
2. Configure local LLM service or API access
3. To start the web server run
   
       python webapp/app.py
5. Access the game interface at `http://127.0.0.1:5001`

## Problem

In computer games players often interact with other entities in the game which are labelled non-player characters (NPCs). NPC dialogue is usually completely pre-scripted which can make the experience of interacting with them repetitive, they also usually lack the ability to dynamically produce dialogue which reflects changes in their environment. Players are often limited to a small number of pre-scripted options to interact with NPCs in dialogue sections, which can make players feel limited in their options for interacting with the characters in a game.

## Project Overview

This is a prototype application for testing intelligent non-player characters (NPCs) that utilize locally-run language models to create dynamic interactions. The system allows players to engage in natural language conversations with NPCs who have their own personalities, memories, knowledge, and contextual awareness. Interaction through the OpenAI API has been included for comparison with the smaller local model.

The system uses LLMs to generate NPC dialogue and to interpret player input dialogue. The system will provide the LLM with additional context about the game environment to simulate an NPC which is embodied within that environment. Using a LLM to interpret the player’s input instead of pre-scripted dialogue will allow the player greater freedom in choice of how they interact with the game environment.

## System Architecture

The application is built on several interconnected components:

1. **Web Interface** - A Flask-based web application that provides the user interface
2. **Knowledge Engine** - Manages game world information and character knowledge
3. **Memory Manager** - Tracks conversations and experiences for each NPC
4. **Executive Planner** - Analyzes player input to determine appropriate responses
5. **Prompt Engine** - Constructs effective prompts for the language model
6. **Ollama Manager** - Handles interactions with the language model services

## Key Components

### Web Application (app.py)

The main Flask application that:

- Serves the game interface
- Handles player interactions via API endpoints
- Manages game state
- Coordinates between all other components
- Supports switching between different AI providers (local Ollama models or OpenAI)
- Enables NPC-to-NPC simulated conversations

### Knowledge Engine (knowledge_engine.py)

Responsible for:

- Storing and retrieving game world information
- Managing what each character knows about the world
- Structuring knowledge into categories (identity, location, events, etc.)
- Searching knowledge based on player queries
- Assessing when knowledge should be incorporated into responses

### Memory Manager (memory_manager.py)

Handles:

- Short-term and long-term memory for NPCs
- Recording interactions between players and NPCs
- Retrieving relevant memories for context
- Summarizing older memories to conserve context space
- Associating memories with specific locations and situations

### Executive Planner (executive.py)

Provides intelligent planning by:

- Analyzing player messages to determine intent
- Identifying when knowledge or memory access is needed
- Classifying message types (greeting, question, command, etc.)
- Determining appropriate response strategies
- Prioritizing information access based on conversation context

### Prompt Engine (prompt_engine.py)

Creates effective prompts by:

- Constructing NPC personalities based on character files
- Including relevant memories and knowledge
- Formatting prompts optimally for the chosen language model
- Supporting different prompt styles for different interaction types
- Handling NPC-to-NPC conversation prompts

### Ollama Manager (ollama_manager.py)

Manages the language model integration:

- Handling requests to local Ollama or OpenAI models
- Streaming responses for natural conversation flow
- Processing and cleaning model outputs
- Managing model selection and configuration
- Supporting different model formats and requirements

## Game Mechanics

### Character System

NPCs are defined in JSON files with:

- Basic identity information
- Personality traits and speech patterns
- Knowledge boundaries
- Background details
- Relationships with other characters

### Knowledge System

The game maintains detailed knowledge structured by:

- Entity types (player, NPC, location, event, item, faction, lore, quest)
- Categories within each entity type
- Confidence ratings for different information pieces
- First-learned and last-updated timestamps
- Information sources

### Memory System

Characters remember interactions through:

- Short-term memories (recent conversations)
- Long-term memories
- Memory indexing by relevance and importance
- Contextual retrieval based on conversation needs

### Conversation Simulation

The system supports:

- Player-to-NPC conversations
- NPC-to-NPC simulated conversations
- Knowledge transfer between characters
- Natural language understanding and generation
- Context-aware responses



### Extending the Game

The system supports:

- Adding new characters by creating JSON definition files
- Expanding game knowledge through the knowledge base
- Creating new locations and scenarios
- Implementing quest systems through the existing framework

## Project Structure

```text
root/
├── README.md                  # Project overview
├── DOCUMENTATION.md           # This comprehensive documentation
├── reqs.txt                   # Required Python packages
├── characters/                # Character definition files
│   ├── blacksmith.json        # Character: Rogar the blacksmith
│   ├── mysterious_stranger.json # Character: Valen the mysterious stranger
│   ├── tavernkeeper.json      # Character: Greta the tavernkeeper
│   └── village_elder.json     # Character: Elder Marlow
├── data/                      # Game data storage
│   ├── game_knowledge.json    # World knowledge database
│   └── memories.json          # NPC memory storage
└── webapp/                    # Application code
    ├── app.py                 # Main Flask application
    ├── executive.py           # Executive planning system
    ├── knowledge_engine.py    # Knowledge management system
    ├── memory_manager.py      # Memory tracking system
    ├── ollama_manager.py      # LLM service integration
    ├── prompt_engine.py       # Prompt construction system
    └── templates/             # Web interface templates
        └── game.html          # Main game interface
```

## Key Design Features

### Dependency Injection Pattern

The code uses dependency injection to share class instances:

- Knowledge Engine, Memory Manager, and Prompt Engine are instantiated in app.py
- These instances are passed to each other as needed
- This allows for better testing, flexibility, and less coupling

### Memory Structure

The memory system has three levels:

1. **Short-term memory**: Recent interactions, stored as complete conversations
2. **Long-term memory**: Older complete conversations moved from short-term memory when it reaches capacity
3. **Summary memory**: Condensed representation created when long-term memory reaches capacity, replacing all long-term memories

### Knowledge Hierarchy

The knowledge base is structured as:

- Global knowledge (available to all characters)
- Character-specific knowledge (unique to each NPC)
- Entity types (player, NPC, location, etc.)
- Categories within entities (identity, background, abilities, etc.)

### Prompt Construction

Prompts include:

1. Character personality and background
2. Relevant memories of past interactions
3. Knowledge needed for the current conversation
4. Context about the current situation
5. Player's input or other NPC's message

## Advanced Features

### NPC-to-NPC Simulation

The system can simulate conversations between NPCs:

- Characters exchange information based on their knowledge
- Memory is updated for both participants
- Knowledge can spread naturally through NPC interactions
- Player can observe these interactions in real-time

### Knowledge Assessment

The system intelligently determines:

- When knowledge access is needed for a response
- Which specific knowledge domains are relevant
- If a character should know this information
- How to integrate knowledge naturally into responses

### Executive Planning

The executive planner makes meta-decisions:

- Classifies input messages by type and intent
- Decides when LLM processing is needed vs. pattern matching
- Prioritizes different processing strategies
- Formulates knowledge queries when information is needed

## Future Development

Potential areas for enhancement:

1. Expanded quest system
2. Emotional state tracking for NPCs
3. Dynamic world events and changes
4. More complex NPC relationships and interactions
5. Enhanced knowledge retrieval mechanisms
6. Support for more language models
7. Integration with graphics or audio elements

---

This documentation provides an overview of the system architecture, components, and functionality. Refer to the individual code files for more detailed implementation information
