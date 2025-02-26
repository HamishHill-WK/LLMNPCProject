# npc_manager.py - NPC dialogue and memory management
import json
import os
import time
from ollama_service import OllamaLLMService

class NPCManager:
    def __init__(self):
        self.characters = self.load_character_profiles()
        # Initialize Ollama service with appropriate model
        self.llm_service = self._initialize_llm_service()
        self.memory_keeper = MemorySystem(self.llm_service)
        self.cache = ResponseCache()
    
    def _initialize_llm_service(self):
        """Initialize the LLM service based on environment settings"""
        # Get settings from environment variables or use defaults
        model_name = os.environ.get("OLLAMA_MODEL", "llama2")
        ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
        ollama_port = os.environ.get("OLLAMA_PORT", "11434")
        base_url = f"http://{ollama_host}:{ollama_port}"
        
        print(f"Initializing Ollama service with model {model_name} at {base_url}")
        
        try:
            # Initialize Ollama service
            service = OllamaLLMService(model_name=model_name, base_url=base_url)
            
            # Get model info for logging
            model_info = service.get_model_info()
            print(f"Successfully connected to Ollama with model: {model_info['model_name']}")
            
            return service
        except Exception as e:
            print(f"Error initializing Ollama service: {e}")
            print("Falling back to local models if available...")
            
            # Fallback to local transformers if Ollama fails
            try:
                from llm_service import LocalLLMService
                return LocalLLMService()
            except Exception as fallback_error:
                print(f"Error initializing fallback service: {fallback_error}")
                raise RuntimeError("Could not initialize any LLM service. Please check your configuration.")
    
    def load_character_profiles(self):
        characters = {}
        if not os.path.exists('characters/'):
            os.makedirs('characters/')
            self._create_sample_character()
            
        character_files = os.listdir('characters/')
        for file in character_files:
            if file.endswith('.json'):
                with open(f'characters/{file}', 'r') as f:
                    character_data = json.load(f)
                    characters[character_data['character_id']] = character_data
        return characters
    
    def _create_sample_character(self):
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
            "knowledge_boundaries": [
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
            },
            "system_prompt": "You are role-playing as Greta, the tavernkeeper in a medieval fantasy setting. You must maintain your established personality traits and background in all interactions. Never break character. Your responses should reflect your gruff nature and military background. Keep your responses concise and practical, focused on the immediate situation and your tavern business. Never use modern language or references."
        }
        
        with open('characters/tavernkeeper.json', 'w') as f:
            json.dump(sample_character, f, indent=2)
    
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
        # Use streaming for better UX with Ollama's generation
        response = self.llm_service.generate_response(
            prompt, 
            max_new_tokens=100,  # Keep responses shorter for local models
            temperature=0.5,     # Lower temperature for more predictable outputs
            stream=True          # Stream response for better UX
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
    
    def validate_response_consistency(self, character, response, game_state):
        # Check for personality trait consistency
        trait_consistency = self.check_trait_alignment(character['core_traits'], response)
        
        # Check for speech pattern consistency
        speech_consistency = self.check_speech_pattern(character['speech_pattern'], response)
        
        # If response seems inconsistent, regenerate or modify
        if trait_consistency < 0.7 or speech_consistency < 0.7:
            # Option 1: Add more character context and regenerate
            enhanced_prompt = self.construct_enhanced_prompt(character, response, game_state)
            return self.llm_service.generate_response(enhanced_prompt)
        
        return response

    def check_trait_alignment(self, traits, response):
        # In a real implementation, this could use:
        # 1. Another LLM call to analyze alignment
        # 2. Keyword/sentiment matching
        # 3. Embedding similarity to exemplar responses
        
        # Simplified version for prototype
        trait_keywords = {
            "gruff": ["direct", "blunt", "terse", "short", "grumble"],
            "fair": ["honest", "equal", "fair", "just", "reasonable"],
            "efficient": ["quick", "efficient", "prompt", "direct", "straightforward"],
            "protective": ["watch", "careful", "protect", "safe", "guard"]
        }
        
        score = 0
        for trait, keywords in trait_keywords.items():
            if any(keyword in response.lower() for keyword in keywords):
                score += 1
        
        return score / len(trait_keywords)

    def check_speech_pattern(self, speech_pattern, response):
        # Simple keyword matching for speech pattern consistency
        speech_pattern_keywords = {
            "Short sentences": lambda r: sum(len(s.split()) < 8 for s in r.split('.')) / max(1, len(r.split('.'))),
            "Northern dialect": lambda r: ('aye' in r.lower() or 'nay' in r.lower()),
            "military metaphors": lambda r: any(word in r.lower() for word in ['battle', 'fight', 'soldier', 'enemy', 'war'])
        }
        
        score = 0
        checks = 0
        
        for pattern, check_func in speech_pattern_keywords.items():
            if pattern.lower() in speech_pattern.lower():
                score += check_func(response)
                checks += 1
        
        return score / max(1, checks)

    def construct_enhanced_prompt(self, character, original_response, game_state):
        # Create a more detailed prompt with explicit consistency guidance
        prompt = f"""
        Your previous response as {character['name']} was not fully consistent with the character's personality.
        
        Character traits to emphasize:
        {', '.join(character['core_traits'])}
        
        Speech pattern to maintain:
        {character['speech_pattern']}
        
        Your previous response: "{original_response}"
        
        Please revise to better match {character['name']}'s personality and speech pattern.
        """
        return prompt


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


class MemorySystem:
    def __init__(self, llm_service=None):
        self.memory_file = 'data/memories.json'
        self.memories = self.load_memories()
        self.max_memories = 10  # Keep last N interactions per NPC
        self.llm_service = llm_service
    
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
                'inventory': game_state['inventory'].copy() if 'inventory' in game_state else [],
                'timestamp': self.get_timestamp()
            }
        })
        
        # Trim to keep only most recent N memories
        if len(self.memories[npc_id]) > self.max_memories:
            self.memories[npc_id] = self.memories[npc_id][-self.max_memories:]
        
        self.save_memories()
        
        # If we have accumulated more than 20 memories, trigger summarization
        if len(self.memories[npc_id]) >= 20 and self.llm_service:
            self.summarize_memories(npc_id)
    
    def get_timestamp(self):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def summarize_memories(self, npc_id):
        """Periodically summarize older memories to prevent context window overflow"""
        if npc_id not in self.memories or len(self.memories[npc_id]) < 20:
            return
        
        # Group memories to summarize (e.g., memories 10-20)
        memories_to_summarize = self.memories[npc_id][10:20]
        
        # Create a summary prompt
        summary_prompt = f"""
        Summarize the following conversation between the player and {npc_id} into 1-2 key points:
        
        {"".join([f"Player: {m['player_message']}\n{npc_id.capitalize()}: {m['response']}\n" for m in memories_to_summarize])}
        """
        
        # Generate summary using LLM
        summary = self.llm_service.generate_response(summary_prompt)
        
        # Replace the detailed memories with the summary
        summary_memory = {
            'player_message': '[SUMMARY]',
            'response': summary,
            'game_state': memories_to_summarize[-1]['game_state'],
            'is_summary': True
        }
        
        # Replace multiple memories with single summary
        self.memories[npc_id] = self.memories[npc_id][:10] + [summary_memory] + self.memories[npc_id][20:]
        self.save_memories()

    def rank_memory_importance(self, npc_id, memory):
        """Assign importance score to new memories for retention decisions"""
        if not self.llm_service:
            return 5  # Default medium importance if no LLM service
            
        # Factors that might make a memory important:
        # - Contains player promises or commitments
        # - Reveals player background information
        # - Contains emotional exchanges
        # - Mentions key game items or characters
        # - Involves quest information
        
        importance_prompt = f"""
        On a scale of 1-10, how important is this exchange to remember for future interactions?
        Consider if it contains promises, personal revelations, emotional moments, or plot-relevant information.
        
        Player: {memory['player_message']}
        {npc_id.capitalize()}: {memory['response']}
        
        Respond with just a number 1-10:
        """
        
        try:
            importance = int(self.llm_service.generate_response(importance_prompt).strip())
            return min(max(importance, 1), 10)  # Ensure between 1-10
        except:
            return 5  # Default medium importance