# prompt_engine.py
import json
import os

# Initialize character data
def load_characters():
    """Load character data from the characters directory"""
    characters = {}
    
    # Create characters directory if it doesn't exist
    if not os.path.exists('characters'):
        os.makedirs('characters')
        create_sample_character()
    
    # Load character files
    character_files = os.listdir('characters')
    for file in character_files:
        if file.endswith('.json'):
            with open(f'characters/{file}', 'r') as f:
                character_data = json.load(f)
                characters[character_data['character_id']] = character_data
    
    return characters

def create_sample_character():
    """Create a sample character if none exist"""
    sample_character = {
        "character_id": "tavernkeeper",
        "name": "Greta",
        "core_traits": [
            "gruff but fair",
            "efficient",
            "protective of establishment",
            "values honesty"
        ],
        "background": "Former soldier who fought in the Northern Wars. Runs the tavern for 15 years since retiring from the army. Lost family during the war and considers the tavern patrons her new family.",
        "speech_pattern": "Short sentences. Northern dialect. Uses 'aye' and 'nay'. Rarely uses pleasantries or small talk. Often uses metaphors related to battle or survival.",
        "knowledge": [
            "Knows local town gossip and politics",
            "Familiar with basic regional history and trade routes",
            "Understands military tactics and weapons",
            "No knowledge of magic or distant kingdoms"
        ],
        "goals": [
            "Keep tavern profitable and respected",
            "Protect regular customers from trouble",
            "Maintain order in her establishment",
            "Avoid entanglements with nobility or officials"
        ],
        "relationships": {
            "village_elder": "Respectful but cautious",
            "blacksmith": "Good friends and drinking buddies",
            "mysterious_stranger": "Deeply suspicious and watchful"
        }
    }
    
    with open('characters/tavernkeeper.json', 'w') as f:
        json.dump(sample_character, f, indent=2)

# Load characters once when module is imported
characters = load_characters()

# Memory management
class MemoryManager:
    def __init__(self, max_short_term=10):
        """Initialize memory manager with specified max short-term memories"""
        self.memories = {}
        self.max_short_term = max_short_term
        
        # Create data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Load existing memories if available
        if os.path.exists('data/memories.json'):
            with open('data/memories.json', 'r') as f:
                self.memories = json.load(f)
    
    def save_memories(self):
        """Save memories to disk"""
        with open('data/memories.json', 'w') as f:
            json.dump(self.memories, f, indent=2)
    
    def add_interaction(self, character_id, player_message, character_response, game_state):
        """Add a new interaction to a character's memory"""
        # Initialize character memory if not exists
        if character_id not in self.memories:
            self.memories[character_id] = {
                "short_term": [],
                "long_term": []
            }
        
        # Add to short-term memory
        memory_item = {
            "player_message": player_message,
            "character_response": character_response,
            "game_state": game_state.copy()
        }
        
        self.memories[character_id]["short_term"].append(memory_item)
        
        # If short-term memory exceeds limit, move oldest to long-term memory
        if len(self.memories[character_id]["short_term"]) > self.max_short_term:
            # For now, just move the oldest memory directly to long-term
            # In a more advanced system, you would summarize a batch of memories
            oldest_memory = self.memories[character_id]["short_term"].pop(0)
            self.memories[character_id]["long_term"].append(oldest_memory)
        
        # Save updated memories
        self.save_memories()
    
    def get_character_memory(self, character_id):
        """Get a character's memory formatted for prompt context"""
        if character_id not in self.memories:
            return "No previous interactions."
        
        memory_text = []
        
        # Add short-term memories
        memory_text.append("--- RECENT INTERACTIONS ---")
        for memory in self.memories[character_id]["short_term"]:
            memory_text.append(f"Player: {memory['player_message']}")
            memory_text.append(f"{characters[character_id]['name']}: {memory['character_response']}")
        
        # Add long-term memories if they exist
        if self.memories[character_id]["long_term"]:
            memory_text.append("\n--- OLDER MEMORIES ---")
            for memory in self.memories[character_id]["long_term"][-3:]:  # Only include last 3 long-term memories
                memory_text.append(f"Player: {memory['player_message']}")
                memory_text.append(f"{characters[character_id]['name']}: {memory['character_response']}")
        
        return "\n".join(memory_text)

# Initialize memory manager
memory_manager = MemoryManager()

def construct_npc_prompt(character_id, player_input, game_state):
    """Construct a prompt for the NPC based on character data and memory"""
    if character_id not in characters:
        return "Error: Character not found."
    
    character = characters[character_id]
    print(character)
    # Get character memory
    memory_context = memory_manager.get_character_memory(character_id)
        # Handle knowledge section safely
    knowledge_section = ""
    if 'knowledge' in characters:
        if isinstance(character['knowledge'], list):
            knowledge_section = ', '.join(character['knowledge'])
        else:
            knowledge_section = str(character['knowledge'])
    
    # Construct the prompt
    prompt = f"""You are roleplaying as {character['name']}, a character in a text adventure game.

CHARACTER TRAITS:
- {', '.join(character['core_traits'])}

BACKGROUND:
{character['background']}

SPEECH PATTERN:
{character['speech_pattern']}

KNOWLEDGE:
{knowledge_section}

CURRENT SITUATION:
- Location: {game_state['current_location']}
- Player has: {', '.join(game_state['inventory']) if game_state['inventory'] else 'nothing notable'}

PREVIOUS INTERACTIONS:
{memory_context}

The player says to you: "{player_input}"

Respond in character as {character['name']}, using your established speech pattern and personality. Keep your response brief (1-3 sentences).\n
Give your response in the following format:
Dialogue: "Your response here"
Actions: Describe any actions or reactions here
"""
    
    return prompt

def add_system_prompt(data, game_state=None):
    """Add the appropriate system prompt to the data based on the current NPC
    
    Args:
        data: Dictionary containing 'prompt' key with player message
        game_state: Optional game state dictionary. If None, uses default state
    
    Returns:
        The constructed NPC prompt
    """
    if game_state is None:
        game_state = {
            'current_location': 'tavern',
            'current_npc': 'blacksmith',
            'inventory': []
        }
    
    character_id = game_state['current_npc']
    player_input = data['prompt']
    
    return construct_npc_prompt(character_id, player_input, game_state)

def record_interaction(character_id, player_message, character_response, game_state):
    """Record an interaction in the character's memory"""
    memory_manager.add_interaction(character_id, player_message, character_response, game_state)
