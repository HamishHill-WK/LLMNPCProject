import time
import json
import random
from typing import Dict, List, Tuple, Optional
import asyncio

# Mock implementation of OllamaLLMService to simulate inference times and responses
class MockOllamaLLMService:
    def __init__(self, model_name="llama2", base_url="http://localhost:11434", 
                 avg_response_time=0.8, variation=0.3):
        self.model_name = model_name
        self.base_url = base_url
        self.avg_response_time = avg_response_time  # Average time in seconds
        self.variation = variation  # Variation in seconds to randomize response time
        print(f"Initialized mock Ollama service with model {model_name}")
        
    def get_model_info(self):
        return {"model_name": self.model_name}
    
    def generate_response(self, prompt, max_new_tokens=100, temperature=0.5, stream=False):
        # Simulate thinking time
        response_time = self.avg_response_time + random.uniform(-self.variation, self.variation)
        time.sleep(response_time)
        
        # Generate a mock response
        characters = {
            "tavernkeeper": {
                "greetings": ["Aye, what can I get ya?", "Welcome to my tavern, traveler.", "Need a drink?"],
                "questions": ["Where ya from?", "Stayin' long?", "Heard any news from the road?"],
                "responses": ["Interesting...", "Aye, I've heard similar tales.", "That'll be two copper."]
            },
            "blacksmith": {
                "greetings": ["Need something forged?", "Welcome to my forge.", "What brings you here?"],
                "questions": ["Need repairs?", "That sword needs sharpening?", "Looking for new gear?"],
                "responses": ["I can craft that.", "That'll take some time to make.", "Fine steel you've got there."]
            },
            "merchant": {
                "greetings": ["Wares for sale!", "Looking to buy?", "Best prices in town!"],
                "questions": ["Interested in this fine cloth?", "Need supplies for your journey?", "Gold or trade?"],
                "responses": ["A fine choice!", "I can offer a discount.", "That's rare stock you're looking at."]
            }
        }
        
        # Try to identify which character might be talking
        character_id = None
        for char_id in characters.keys():
            if char_id.lower() in prompt.lower():
                character_id = char_id
                break
        
        if not character_id:
            character_id = random.choice(list(characters.keys()))
        
        char_data = characters[character_id]
        
        # Determine response type based on prompt content
        if "hello" in prompt.lower() or "greet" in prompt.lower():
            response = random.choice(char_data["greetings"])
        elif "?" in prompt:
            response = random.choice(char_data["responses"])
        else:
            response = random.choice(char_data["questions"])
            
        return response

class MemorySystem:
    def __init__(self):
        self.memories = {}
    
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
            'game_state': game_state
        })


class NPCConversationSimulator:
    def __init__(self, characters=None):
        self.llm_service = MockOllamaLLMService()
        self.memory_system = MemorySystem()
        self.characters = characters or self._load_default_characters()
        
    def _load_default_characters(self):
        return {
            "tavernkeeper": {
                "name": "Greta",
                "core_traits": ["gruff but fair", "efficient", "protective of establishment"],
                "background": "Former soldier who fought in the Northern Wars. Runs the tavern for 15 years.",
                "speech_pattern": "Short sentences. Northern dialect. Uses 'aye' and 'nay'."
            },
            "blacksmith": {
                "name": "Dorn",
                "core_traits": ["strong", "meticulous", "honest"],
                "background": "Village blacksmith for 20 years. Learned the trade from his father.",
                "speech_pattern": "Direct speech. Technical terms. Deep voice."
            },
            "merchant": {
                "name": "Lysara",
                "core_traits": ["shrewd", "friendly", "well-traveled"],
                "background": "Traveling merchant who visits the village seasonally. Brings exotic goods.",
                "speech_pattern": "Enthusiastic. Uses foreign phrases. Descriptive language."
            }
        }
    
    def construct_prompt(self, character, message, memories, game_state):
        prompt = f"""You are roleplaying as {character['name']}, a character in a text adventure game.

NAME: {character['name']}

CORE TRAITS: {', '.join(character['core_traits'])}

BACKGROUND: {character['background']}

SPEECH PATTERN: {character['speech_pattern']}

{memories}

CURRENT SITUATION:
- Location: {game_state['current_location']}
- Players present: {', '.join(game_state['npcs_present']) if game_state['npcs_present'] else 'none'}

The following character says to you: "{message}"

Respond briefly (1-3 sentences) in character as {character['name']}:"""
        
        return prompt
    
    def get_npc_response(self, npc_id, message, speaker_id, game_state):
        start_time = time.time()
        
        if npc_id not in self.characters:
            return "Error: Character not found.", 0
        
        # Get character profile
        character = self.characters[npc_id]
        
        # Retrieve relevant memories
        memories = self.memory_system.get_memories(npc_id, game_state)
        
        # Construct prompt
        prompt = self.construct_prompt(character, message, memories, game_state)
        
        # Get response from LLM
        print(f"Generating response from {character['name']}...")
        response = self.llm_service.generate_response(
            prompt, 
            max_new_tokens=100,
            temperature=0.5,
            stream=True
        )
        
        # Store this interaction in memory
        self.memory_system.store_interaction(npc_id, message, response, game_state)
        
        elapsed_time = time.time() - start_time
        
        return response, elapsed_time
    
    async def simulate_conversation(self, npc1_id, npc2_id, num_turns=5, initial_message=None, game_state=None):
        """Simulate a conversation between two NPCs for a specified number of turns."""
        if not game_state:
            game_state = {
                "current_location": "Village Square",
                "npcs_present": [npc1_id, npc2_id],
                "time_of_day": "afternoon"
            }
            
        # Set up the conversation with an initial message
        current_speaker = npc1_id
        current_listener = npc2_id
        current_message = initial_message or "Hello there, how are you today?"
        
        conversation_log = []
        total_inference_time = 0
        
        print(f"\n===== Starting conversation between {npc1_id} and {npc2_id} =====\n")
        
        # Log initial message
        conversation_log.append({
            "speaker": current_speaker,
            "message": current_message,
            "inference_time": 0
        })
        
        print(f"{self.characters[current_speaker]['name']}: {current_message}")
        
        # Conduct the conversation for the specified number of turns
        for turn in range(num_turns):
            # Swap speaker and listener
            current_speaker, current_listener = current_listener, current_speaker
            
            # Get response from current speaker
            response, inference_time = self.get_npc_response(
                current_speaker, 
                current_message, 
                current_listener,
                game_state
            )
            
            # Update the current message for the next turn
            current_message = response
            
            # Log the response
            conversation_log.append({
                "speaker": current_speaker,
                "message": response,
                "inference_time": inference_time
            })
            
            total_inference_time += inference_time
            
            print(f"{self.characters[current_speaker]['name']}: {response} ({inference_time:.2f}s)")
            
            # Add a small delay to make the conversation more readable
            await asyncio.sleep(0.5)
        
        # Print summary
        avg_inference_time = total_inference_time / num_turns
        print(f"\n===== Conversation Summary =====")
        print(f"Total turns: {num_turns}")
        print(f"Total inference time: {total_inference_time:.2f} seconds")
        print(f"Average inference time per turn: {avg_inference_time:.2f} seconds")
        
        return conversation_log, total_inference_time, avg_inference_time

    def run_batch_simulations(self, num_simulations=5, turns_per_simulation=5):
        """Run multiple conversation simulations and aggregate the results."""
        character_ids = list(self.characters.keys())
        results = []
        
        for i in range(num_simulations):
            # Select random characters for conversation
            npc1 = random.choice(character_ids)
            npc2 = random.choice([char for char in character_ids if char != npc1])
            
            initial_messages = [
                "Hello there, how are you today?",
                "Have you heard the latest news?",
                "I've been looking for you.",
                "What do you think about the rumors of bandits?",
                "The weather has been strange lately."
            ]
            
            conversation_result = asyncio.run(self.simulate_conversation(
                npc1, 
                npc2, 
                turns_per_simulation,
                random.choice(initial_messages)
            ))
            
            results.append({
                "npc1": npc1,
                "npc2": npc2,
                "log": conversation_result[0],
                "total_time": conversation_result[1],
                "avg_time": conversation_result[2]
            })
            
            # Small delay between simulations
            time.sleep(1)
        
        # Calculate aggregate statistics
        total_inference_times = [result["total_time"] for result in results]
        avg_inference_times = [result["avg_time"] for result in results]
        
        overall_avg = sum(avg_inference_times) / len(avg_inference_times)
        max_avg = max(avg_inference_times)
        min_avg = min(avg_inference_times)
        
        print("\n===== Batch Simulation Summary =====")
        print(f"Number of simulations: {num_simulations}")
        print(f"Turns per simulation: {turns_per_simulation}")
        print(f"Overall average inference time per turn: {overall_avg:.2f} seconds")
        print(f"Min average inference time: {min_avg:.2f} seconds")
        print(f"Max average inference time: {max_avg:.2f} seconds")
        
        return results, overall_avg, min_avg, max_avg


# Example usage - can be run directly or imported as a module
if __name__ == "__main__":
    # Create simulator with default characters
    simulator = NPCConversationSimulator()
    
    # Option 1: Run a single conversation simulation
    asyncio.run(simulator.simulate_conversation("tavernkeeper", "merchant", num_turns=6))
    
    # Option 2: Run batch simulations to get performance metrics
    # simulator.run_batch_simulations(num_simulations=3, turns_per_simulation=4)
    
    # Option 3: Test with different inference time settings
    # fast_simulator = NPCConversationSimulator()
    # fast_simulator.llm_service.avg_response_time = 0.3  # Faster responses
    # asyncio.run(fast_simulator.simulate_conversation("tavernkeeper", "blacksmith", num_turns=4))
