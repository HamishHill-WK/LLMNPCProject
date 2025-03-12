# Memory management
import os
import json 
import re

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
    
    def add_interaction(self, character_id, player_message, character_response, location):
        """Add a new interaction to a character's memory"""
        # Initialize character memory if not exists
        if character_id not in self.memories:
            self.memories[character_id] = {
                "short_term": [],
                "long_term": []
            }
            
        # Extract the chain of thought (content between <think> tags)
        chain_of_thought_match = re.search(r'<think>(.*?)</think>', character_response, re.DOTALL)
        chain_of_thought = chain_of_thought_match.group(1).strip() if chain_of_thought_match else ""

        # Remove the <think> content from the character response
        clean_response = re.sub(r'<think>.*?</think>', '', character_response, flags=re.DOTALL).strip()
        
        # Split the response into dialogue and actions if "Character Actions:" is present
        dialogue = clean_response
        character_actions = ""

        if "Character Actions:" in clean_response:
            parts = clean_response.split("Character Actions:", 1)
            dialogue = parts[0].strip()
            character_actions = parts[1].strip() if len(parts) > 1 else ""
        
        # Add to short-term memory
        memory_item = {
            "memory_id": len(self.memories[character_id]['short_term']) + 1,
            "player_message": player_message,
            "Model Chain of thought" : chain_of_thought,
            "character_response": dialogue,
            "character_actions": character_actions,
            "location": location
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
    
    def get_character_memory(self, characters : dict, character_id):
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

