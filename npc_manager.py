
# npc_manager.py - NPC dialogue and memory management
import json
import os
import time
from llm_service import LocalLLMService

class NPCManager:
    def __init__(self):
        self.characters = self.load_character_profiles()
        # Initialize with a small, low-resource model for development
        self.llm_service = LocalLLMService(
            model_name="unsloth/DeepSeek-R1-Distill-Qwen-7B-Q2_K.gguf", 
            device="cuda" if os.environ.get("USE_GPU", "0") == "1" else "cpu"
        )
        self.memory_keeper = MemorySystem()
        self.cache = ResponseCache()
    
    def load_character_profiles(self):
        characters = {}
        character_files = os.listdir('characters/')
        for file in character_files:
            if file.endswith('.json'):
                with open(f'characters/{file}', 'r') as f:
                    character_data = json.load(f)
                    characters[character_data['character_id']] = character_data
        return characters
    
    def get_npc_response(self, npc_id, player_message, game_state):
        if npc_id not in self.characters:
            return "Error: Character not found."
        
        # Get character profile
        character = self.characters[npc_id]
        
        # Check cache first - for common greetings and responses
        cache_key = f"{npc_id}:{player_message.lower().strip()}"
        cached_response = self.cache.get(cache_key)
        if cached_response:
            print("Using cached response")
            return cached_response
        
        # Retrieve relevant memories
        memories = self.memory_keeper.get_memories(npc_id, game_state)
        
        # Construct prompt with character info and memories
        prompt = self.construct_prompt(character, player_message, memories, game_state)
        
        # Add a progress indicator in the UI
        print(f"Generating response from {character['name']}...")
        
        # Get response from LLM - use lower temperature for more consistent responses
        response = self.llm_service.generate_response(
            prompt, 
            max_new_tokens=100,  # Keep responses shorter for local models
            temperature=0.5      # Lower temperature for more predictable outputs
        )
        
        # Basic response filtering
        response = self.filter_response(response, character)
        
        # Store this interaction in memory
        self.memory_keeper.store_interaction(npc_id, player_message, response, game_state)
        
        # Cache common responses
        if len(player_message) < 20 and player_message.lower() in [
            "hello", "hi", "greetings", "hey", "how are you", 
            "goodbye", "bye", "farewell", "see you", "thanks", "thank you"
        ]:
            self.cache.set(cache_key, response)
        
        return response
    
    def construct_prompt(self, character, player_message, memories, game_state):
        # Optimize prompt for memory-constrained local models
        # Focus on essential character elements and keep context minimal
        
        prompt = f"""You are roleplaying as {character['name']}, a character in a text adventure game.

NAME: {character['name']}

CORE TRAITS: {', '.join(character['core_traits'])}

BACKGROUND: {character['background'][:200]}... (condensed for brevity)

SPEECH PATTERN: {character['speech_pattern']}

{memories[:500] if len(memories) > 500 else memories}

CURRENT SITUATION:
- Location: {game_state['current_location']}
- Player has: {', '.join(game_state['inventory']) if game_state['inventory'] else 'nothing notable'}

The player says to you: "{player_message}"

Respond briefly (1-3 sentences) in character as {character['name']}:"""
        
        return prompt
    
    def filter_response(self, response, character):
        """Apply basic filtering to ensure response quality"""
        # Remove any preamble like "As [character name]" that models often add
        prefixes_to_remove = [
            f"As {character['name']},", 
            f"{character['name']}:",
            "I respond,",
            "I say,",
            "*Roleplaying as",
            "*In character*"
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # If response is too long, truncate to complete sentences
        if len(response) > 300:
            sentences = response.split('.')
            truncated = []
            total_length = 0
            
            for sentence in sentences:
                if total_length + len(sentence) < 300:
                    truncated.append(sentence)
                    total_length += len(sentence) + 1  # +1 for the period
                else:
                    break
            
            response = '.'.join(truncated) + '.'
        
        return response

class ResponseCache:
    """Simple cache for common NPC responses to improve performance"""
    def __init__(self, max_size=100, ttl=3600):  # 1 hour TTL
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key):
        """Get a cached response if available and not expired"""
        if key in self.cache:
            # Check if entry has expired
            if time.time() - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            return self.cache[key]
        return None
    
    def set(self, key, value):
        """Add a response to the cache"""
        # If cache is full, remove oldest entry
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = value
        self.timestamps[key] = time.time()

### C. Memory System

# Part of npc_manager.py
class MemorySystem:
    def __init__(self):
        self.memory_file = 'data/memories.json'
        self.memories = self.load_memories()
        self.max_memories = 10  # Keep last N interactions per NPC
    
    def load_memories(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_memories(self):
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, 'w') as f:
            json.dump(self.memories, f)
    
    def get_memories(self, npc_id, game_state):
        if npc_id not in self.memories:
            return "No previous interactions."
        
        memory_strings = []
        for memory in self.memories[npc_id][-5:]:  # Last 5 memories
            memory_strings.append(f"Player: {memory['player_message']}\n{npc_id.capitalize()}: {memory['response']}")
        
        return "\n\n".join(memory_strings)
    
    def store_interaction(self, npc_id, player_message, response, game_state):
        if npc_id not in self.memories:
            self.memories[npc_id] = []
        
        self.memories[npc_id].append({
            'player_message': player_message,
            'response': response,
            'game_state': {
                'location': game_state['current_location'],
                'inventory': game_state['inventory'].copy(),
                'timestamp': self.get_timestamp()
            }
        })
        
        # Trim to keep only most recent N memories
        if len(self.memories[npc_id]) > self.max_memories:
            self.memories[npc_id] = self.memories[npc_id][-self.max_memories:]
        
        self.save_memories()
    
    def get_timestamp(self):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

