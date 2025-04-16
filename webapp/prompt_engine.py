# prompt_engine.py
import json
import os
import memory_manager
import knowledge_engine as ke

class Prompt_Engine:
    def __init__(self, knowledge_engine: ke.KnowledgeEngine, memory_manager: memory_manager.MemoryManager):
        self.characters = self.load_characters()
        self.knowledge_engine = knowledge_engine
        self.memory_manager = memory_manager
        
# Initialize character data
    def load_characters(self):
        """Load character data from the characters directory"""
        characters = {}       
        # Load character files
        character_files = os.listdir('characters')
        for file in character_files:
            if file.endswith('.json'):
                with open(f'characters/{file}', 'r') as f:
                    character_data = json.load(f)
                    characters[character_data['character_id']] = character_data
        
        return characters

    def construct_npc_prompt(self, character_id, player_input, game_state, data):
        """Construct a prompt for the NPC based on character data and memory"""
        if character_id not in self.characters:
            return "Error: Character not found."
        character = self.characters[character_id]
        # Get character memory
        memory_context = "No previous interactions."
        if self.memory_manager is not None:
            memory_context = self.memory_manager.get_character_memory(game_state['all_characters'], character_id)

        knowledge_sections = []
        player_knowledge = self.knowledge_engine.get_player_knowledge(character_id)
        if player_knowledge and player_knowledge != "You don't know much about the player yet.":
            knowledge_sections.append(f"ABOUT THE PLAYER:\n{player_knowledge}")

        location_knowledge = self.knowledge_engine.get_entity_knowledge(character_id, "location", game_state['current_location'])
        if location_knowledge and location_knowledge != f"You don't know much about {game_state['current_location']}.":
            knowledge_sections.append(f"ABOUT {game_state['current_location'].upper()}:\n{location_knowledge}")
                    
        if 'relevant_knowledge' in data:
            knowledge_sections.append(f"RELEVANT KNOWLEDGE:\n{data['relevant_knowledge']}")
        
        # 3. Get knowledge about other NPCs mentioned in recent conversation
        old_memory = self.memory_manager.get_memory_summary(character_id)
        conversation_text = memory_context #+ " " + player_input
        for npc_id in game_state['all_characters']:
            if npc_id != character_id and npc_id in conversation_text:
                # NPC is mentioned in conversation
                npc_data = self.characters.get(npc_id, {})
                npc_name = npc_data.get('name', npc_id)
                npc_knowledge = self.knowledge_engine.get_entity_knowledge(character_id, "npc", npc_name)
                if npc_knowledge and npc_knowledge != f"You don't know much about {npc_name}.":
                    knowledge_sections.append(f"ABOUT {npc_name.upper()}:\n{npc_knowledge}")

        knowledge_section = ""
        if 'knowledge' in character:
            if isinstance(character['knowledge'], list):
                knowledge_section = ', '.join(character['knowledge'])
            else:
                knowledge_section = str(character['knowledge'])
                        
        # Combine all knowledge sections
        combined_knowledge = "\n\n".join(knowledge_sections) if knowledge_sections else ""
    
        # Construct the prompt
        prompt = f"""<system>You are roleplaying as {character['name']}, a character in a text adventure game.

    <character_profile>
    CHARACTER TRAITS:
    - {', '.join(character['core_traits'])}

    BACKGROUND:
    {character['background']}

    SPEECH PATTERN:
    {character['speech_pattern']}

    KNOWLEDGE:
    {knowledge_section}
    {combined_knowledge}
    </character_profile>

    <older_interaction_summary>
    {old_memory}
    </older_interaction_summary>

    <previous_interactions>
    {memory_context}
    </previous_interactions>

    <system>
    CURRENT SITUATION:
    - Location: {game_state['current_location']}

    If the player asks a question or makes a request, you should respond in character, building on the previous interactions and knowledge boundaries of {character['name']}.
    If the player says goodbye or otherwise ends the conversation, you should end the interaction naturally.
    Ask the player questions to move the conversation forward and to learn about them. Build on previous interactions and knowledge to progress the conversation naturally..
    Respond in character as {character['name']}, using your established knowledge, speech pattern and personality. Don't write more than a paragraph.
    Previous interactions have been included for context. Try not to repeat previous responses unless it is absolutely necessary.
    Older interactions have been summarized for brevity.
    Do not acknowledge that you are a character in a game or break the fourth wall.
    Give your reponse in the format provided below in the <character_response> tags.
    <player_message>
    Most recent Player message: {player_input}
    </player_message>
    <character_response>
    {character['name']}: "[Your response here]"
    </character_response>
    Do not write anything after the character response as this will be ignored by the system.
    </system>
    """
        
        return prompt

    def construct_inter_npc_prompt(self, speaker_id, speaker_input, simulation_state, mem_manager: memory_manager.MemoryManager):
        """Construct a prompt for NPC-to-NPC interaction with full knowledge integration"""
        # Determine which NPC is listening and which is speaking
        listener_npc = simulation_state['npc_A'] if speaker_id == 'npc_B' else simulation_state['npc_B']
        speaker_npc = simulation_state['npc_A'] if speaker_id == 'npc_A' else simulation_state['npc_B']
        speaker_npc_obj = self.characters.get(speaker_npc, {})
        speaker_name = speaker_npc_obj.get('name', speaker_npc)

        initial_prompt = simulation_state['initial_prompt']
        if listener_npc not in self.characters:
            return "Error: Character not found."
        
        character = self.characters[listener_npc]
        
        # Get character memory
        memory_context = "No previous interactions."
        if mem_manager is not None:
            memory_context = mem_manager.get_character_memory(simulation_state['all_characters'], listener_npc)
        
        # Retrieve long-term memory summary
        old_memory = mem_manager.get_memory_summary(listener_npc)
        
        # Build knowledge sections similar to player-NPC interactions
        knowledge_sections = []
        
        # Add location knowledge
        location_knowledge = self.knowledge_engine.get_entity_knowledge(listener_npc, "location", simulation_state['current_location'])
        if location_knowledge and location_knowledge != f"You don't know much about {simulation_state['current_location']}.":
            knowledge_sections.append(f"ABOUT {simulation_state['current_location'].upper()}:\n{location_knowledge}")
        
        # Most importantly, add knowledge about the speaking NPC
        speaker_knowledge = self.knowledge_engine.get_entity_knowledge(listener_npc, "npc", speaker_name)
        if speaker_knowledge and speaker_knowledge != f"You don't know much about {speaker_name}.":
            knowledge_sections.append(f"ABOUT {speaker_name.upper()}:\n{speaker_knowledge}")
        
        # Check if any other NPCs are mentioned in the conversation
        conversation_text = memory_context + " " + speaker_input
        for npc_id in simulation_state['all_characters']:
            if npc_id != listener_npc and npc_id != speaker_npc and npc_id in conversation_text:
                # Another NPC is mentioned
                npc_data = self.characters.get(npc_id, {})
                npc_name = npc_data.get('name', npc_id)
                npc_knowledge = self.knowledge_engine.get_entity_knowledge(listener_npc, "npc", npc_name)
                if npc_knowledge and npc_knowledge != f"You don't know much about {npc_name}.":
                    knowledge_sections.append(f"ABOUT {npc_name.upper()}:\n{npc_knowledge}")
        
        # Get character's general knowledge
        knowledge_section = ""
        if 'knowledge' in character:
            if isinstance(character['knowledge'], list):
                knowledge_section = ', '.join(character['knowledge'])
            else:
                knowledge_section = str(character['knowledge'])
        
        # Combine all knowledge sections
        combined_knowledge = "\n\n".join(knowledge_sections) if knowledge_sections else ""
        
        # Construct the enhanced prompt
        prompt = f"""<system>You are roleplaying as {character['name']}, a character in a text adventure game.

<character_profile>
CHARACTER TRAITS:
- {', '.join(character['core_traits'])}

BACKGROUND:
{character['background']}

SPEECH PATTERN:
{character['speech_pattern']}

KNOWLEDGE:
{knowledge_section}
{combined_knowledge}
</character_profile>

<older_interaction_summary>
{old_memory}
</older_interaction_summary>

<previous_interactions>
{memory_context}
</previous_interactions>

<other_character_message>
{speaker_name}: {speaker_input}
</other_character_message>

<system>
CURRENT SITUATION:
- Location: {simulation_state['current_location']}

You are currently in a conversation with {speaker_name}. The initial context was:
{initial_prompt}

If {speaker_name} asks a question or makes a request, you should respond in character, building on your relationship and previous interactions.
Respond naturally as {character['name']}, using your established speech pattern and personality. Don't write more than a paragraph.
Your response should reflect your character's attitude toward {speaker_name} based on your relationship and history.
Previous interactions have been included for context. Try not to repeat previous responses unless it is absolutely necessary.
Older interactions have been summarized for brevity.
You can use <think> tags to write your thought process, which will not be part of your actual response.
</system>

<character_response>
{character['name']}:
</character_response>
"""
    
        return prompt
    
    def add_system_prompt(self, data, game_state=None, mem_manager=None):
        #Add the appropriate system prompt to the data based on the current NPC
        if 'current_speaker' in game_state:
            return self.construct_inter_npc_prompt(game_state['current_speaker'], data['prompt'], game_state, mem_manager)
        
        character_id = game_state['current_npc']
        player_input = data['prompt']
        
        knowledge_analysis = data.get("knowledge_analysis", {})
        if knowledge_analysis.get("knowledge_required", False) and knowledge_analysis.get("requires_memory", False):
            return self.construct_npc_prompt(character_id, player_input, game_state)
        else:
            
            return self.construct_npc_prompt(character_id, player_input, game_state, data)