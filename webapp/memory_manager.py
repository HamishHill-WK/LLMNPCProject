# Memory management
import os
import json 
import re
import datetime
import ollama_manager as om

class MemoryManager:
    def __init__(self, max_short_term=3, max_long_term = 3, characters={}, knowledge_engine=None, ollama=None):
        """Initialize memory manager with specified max short-term memories"""
        self.memories = {}
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
        self.characters = characters
        self.knowledge_engine = knowledge_engine
        self.ollama = ollama
        
        # Create data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Load existing memories if available
        if os.path.exists('data/memories.json'):
            with open('data/memories.json', 'r') as f:
                self.memories = json.load(f)
    
    def save_characters(self, characters):
        """Load characters into memory manager"""
        self.characters = characters
    
    def set_ollama(self, ollama):
        """Set the ollama manager for this memory manager"""
        self.ollama = ollama
    
    def save_memories(self):
        """Save memories to disk"""
        with open('data/memories.json', 'w') as f:
            json.dump(self.memories, f, indent=2)
    
    def add_interaction(self, character_id, other_id, other_message, character_response, chain_of_thought, location):
        """Add a new interaction to a character's memory"""
        # Initialize character memory if not exists
        if character_id not in self.memories:
            self.memories[character_id] = {
                "short_term": [],
                "long_term": []
            }
        
        # Remove <character_response> tags if present
        dialogue = character_response.replace('<character_response>', '').replace('</character_response>', '').strip()
        # Only try to remove character name if it exists in the characters dictionary
        if character_id in self.characters and 'name' in self.characters[character_id]:
            dialogue = dialogue.replace(f"{self.characters[character_id]['name']}: ", "").strip()
    
        # Split the response into dialogue and actions if "Character Actions:" is present
        character_actions = ""
        if "Character Actions:" in dialogue:
            parts = dialogue.split("Character Actions:", 1)
            dialogue = parts[0].strip()
            character_actions = parts[1].strip() if len(parts) > 1 else ""
        
        # Add to short-term memory
        memory_item = {
            "memory_id": len(self.memories[character_id]['short_term']) + 1,
            "other_id" : other_id,
            "other_message": other_message,
            "Model Chain of thought" : chain_of_thought,
            "character_response": dialogue,
            "character_actions": character_actions,
            "location": location
        }
        
        self.memories[character_id]["short_term"].append(memory_item)
        # # If short-term memory exceeds limit, move oldest to long-term memory
        if len(self.memories[character_id]["short_term"]) > self.max_short_term:
            # For now, just move the oldest memory directly to long-term
            # In a more advanced system, you would summarize a batch of memories
            oldest_memory = self.memories[character_id]["short_term"].pop(0)
            
            if "long_term" not in self.memories[character_id]:
                self.memories[character_id]["long_term"] = []
            
            self.memories[character_id]["long_term"].append(oldest_memory)
            if len (self.memories[character_id]["long_term"]) > self.max_long_term:
                self.memories[character_id]["summary"], self.memories[character_id]["summary_cof"] = self.summarize_long_term(character_id)
                self.memories[character_id]["long_term"] = []
        
        # Save updated memories
        self.save_memories()
    
    def get_character_memory(self, characters : dict, character_id):
        """Get a character's memory formatted for prompt context"""
        if character_id not in self.memories:
            return "No previous interactions."
        
        memory_text = []        
        # Add short-term memories
        if 'short_term' in self.memories[character_id]:
            for memory in self.memories[character_id]["short_term"]:
                memory_text.append(f"{memory['other_id']}: {memory['other_message']}")
                memory_text.append(f"{characters[character_id]['name']}: {memory['character_response']}")
            
        if 'long_term' in self.memories[character_id]:
            for memory in self.memories[character_id]["long_term"]:
                memory_text.append(f"{memory['other_id']}: {memory['other_message']}")
                memory_text.append(f"{characters[character_id]['name']}: {memory['character_response']}")
        
        return "\n".join(memory_text)
    
    def get_memory_summary(self, character_id):
        """Get a character's summarized long-term memory"""
        if character_id not in self.memories:
            return ""
        
        if "summary" in self.memories[character_id]:
            return self.memories[character_id]["summary"]
        else:
            return ""
    
    def summarize_long_term(self, character_id):
        """Summarize short-term memory for a character"""
        if character_id not in self.memories:
            return ""
        
        summary = ""
        character_name = self.characters[character_id]['name']
        if 'long_term' in self.memories[character_id]:
            for memory in self.memories[character_id]["long_term"]:
                summary += f"{memory['other_id']}: {memory['other_message']}\n"
                summary += f"{character_name}: {memory['character_response']}\n"
            
        old_summary = ""
        
        if "summary" in self.memories[character_id]:
            old_summary = f"PREVIOUS SUMMARY:\n{self.memories[character_id]["summary"]}"
        
        prompt = ""
        if old_summary != "":
            prompt = f"""
<system>
Summarize the following information into a concise character memory from the point of view of . Integrate the previous summary 
with new interactions to create a comprehensive, updated understanding:

{old_summary}

NEW INTERACTIONS:
{summary}

Create a new comprehensive summary that includes all important information from both sources.
Focus on key facts, relationships, preferences, and character traits revealed.
Resolve any contradictions by favoring newer information.
</system>
        """
        else:
            prompt = f"Summarize this exchange from the point of view of {character_name}:\n{summary}\n         Focus on key facts, relationships, preferences, and character traits revealed."
        
        response = self.ollama.get_response({
            "model": "deepseek-r1:8b",
            "prompt": prompt,
            "stream": False,
            "max_tokens": 150
        }, None, None)
        
        
        #print(f"Memory summary for {character_name}: {response}")

        response, chain_of_thought = self.ollama.clean_response(response)
        
        return response, chain_of_thought