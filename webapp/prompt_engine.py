# prompt_engine.py
import json
import os
import memory_manager
# Initialize character data
def load_characters():
    """Load character data from the characters directory"""
    characters = {}
    
    # Create characters directory if it doesn't exist
    # if not os.path.exists('characters'):
    #     os.makedirs('characters')
    #     create_sample_character()
    
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

def construct_npc_prompt(character_id, player_input, game_state, mem_manager : memory_manager.MemoryManager):
    """Construct a prompt for the NPC based on character data and memory"""
    if character_id not in characters:
        return "Error: Character not found."
    character = characters[character_id]
    # Get character memory
    memory_context = "No previous interactions."
    if mem_manager is not None:
        memory_context = mem_manager.get_character_memory(game_state['all_characters'], character_id)
        
    knowledge_section = ""
    if 'knowledge' in characters:
        if isinstance(character['knowledge'], list):
            knowledge_section = ', '.join(character['knowledge'])
        else:
            knowledge_section = str(character['knowledge'])
    
    # Construct the prompt
    prompt = f"""<system>You are roleplaying as {character['name']}, a character in a text adventure game.
</system>

<character_profile>
CHARACTER TRAITS:
- {', '.join(character['core_traits'])}

BACKGROUND:
{character['background']}

SPEECH PATTERN:
{character['speech_pattern']}

KNOWLEDGE:
{knowledge_section}
</character_profile>

<previous_interactions>
{memory_context}
</previous_interactions>

<player_message>
Player: {player_input}
</player_message>

<system>
CURRENT SITUATION:
- Location: {game_state['current_location']}

The player says to you: "{player_input}"

If the player asks a question or makes a request, you should respond in character based on the previous interactions and knowledge in the text provided above.
If the player says goodbye or otherwise ends the conversation, you should end the interaction naturally.
Ask the player questions to move the conversation forward and to learn about them. Build on previous interactions and progress the conversation naturally.
Respond in character as {character['name']}, using your established speech pattern and personality. Don't write more than a paragraph.

You can use <think> tags to write your thought process, which will not be part of your actual response.
</system>

<character_response>
{character['name']}:
</character_response>
"""
    
    return prompt

def construct_inter_npc_prompt(speaker_id, speaker_input, simulation_state, mem_manager : memory_manager.MemoryManager):
    """Construct a prompt for the NPC based on character data and memory"""
    listener_npc = simulation_state['npc_A'] if speaker_id == 'npc_B' else simulation_state['npc_B']
    speaker_npc = simulation_state['npc_A'] if speaker_id == 'npc_A' else simulation_state['npc_B']

    inital_prompt = simulation_state['initial_prompt']
    if listener_npc not in characters:
        return "Error: Character not found."
    
    character = characters[listener_npc]
    # Get character memory
    memory_context = "No previous interactions."
    if mem_manager is not None:
        memory_context = mem_manager.get_character_memory(simulation_state['all_characters'], listener_npc)
    
    knowledge_section = ""
    if 'knowledge' in characters:
        if isinstance(character['knowledge'], list):
            knowledge_section = ', '.join(character['knowledge'])
        else:
            knowledge_section = str(character['knowledge'])
    
    # Construct the prompt
    prompt = f"""<system>You are roleplaying as {character['name']}, a character in a text adventure game.
</system>

<character_profile>
CHARACTER TRAITS:
- {', '.join(character['core_traits'])}

BACKGROUND:
{character['background']}

SPEECH PATTERN:
{character['speech_pattern']}

KNOWLEDGE:
{knowledge_section}
</character_profile>

<previous_interactions>
{memory_context}
</previous_interactions>

<other_character_message>
{speaker_npc}: {speaker_input}
</other_character_message>

<system>
CURRENT SITUATION:
- Location: {simulation_state['current_location']}

You are currently interacting with {speaker_npc} in a text adventure game. The initial prompt was:
{inital_prompt}

If {speaker_npc} asks a question or makes a request, you should respond in character based on the previous interactions and knowledge appropriate to {character['name']}.
Respond in character as {character['name']}, using your established speech pattern and personality. Don't write more than a paragraph.
You can use <think> tags to write your thought process, which will not be part of your actual response.
</system>

<character_response>
{character['name']}:
</character_response>
"""
    
    return prompt

def construct_memory_context(character_id, mem_manager : memory_manager.MemoryManager):
    """Construct a memory context for a character based on memory data"""
    

def add_system_prompt(data, game_state=None, mem_manager=None):
    #Add the appropriate system prompt to the data based on the current NPC
    if 'current_speaker' in game_state:
        return construct_inter_npc_prompt(game_state['current_speaker'], data['prompt'], game_state, mem_manager)
    
    character_id = game_state['current_npc']
    player_input = data['prompt']
    
    return construct_npc_prompt(character_id, player_input, game_state, mem_manager)
