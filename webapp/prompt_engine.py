# prompt_engine.py
import json
import os
import memory_manager
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


def construct_npc_prompt(character_id, player_input, game_state):
    """Construct a prompt for the NPC based on character data and memory"""
    if character_id not in characters:
        return "Error: Character not found."
    
    character = characters[character_id]
    print(character)
    # Get character memory
   # memory_context = memory_manager.get_character_memory(character_id)
    memory_context = "No previous interactions."
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

If the player asks a question or makes a request, you should respond in character based on the previous interactions and knowledge in the text provided above.
Respond in character as {character['name']}, using your established speech pattern and personality. Keep your response brief (1-3 sentences).\n
Give your response in the following format:
{character['name']} Dialogue output: "Character response here"
Character Actions: Describe any actions or reactions here
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
