#!/usr/bin/env python3
# npc_conversation_tester.py - Test inter-NPC conversations using local LLM (Ollama)

import os
import json
import time
import requests
from collections import deque
import argparse
import logging
from typing import List, Dict, Any, Tuple
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("npc_conversation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NPCConversationTester")

class OllamaLLMService:
    """Service for interacting with local Ollama LLM"""
    
    def __init__(self, model_name="llama2", base_url="http://localhost:11434"):
        """Initialize the Ollama LLM service"""
        self.model_name = model_name
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/generate"
        
        # Test connection
        try:
            self.get_model_info()
            logger.info(f"Successfully connected to Ollama with model: {model_name}")
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            raise ConnectionError(f"Could not connect to Ollama at {base_url}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_info = next((m for m in models if m["name"] == self.model_name), None)
                if model_info:
                    return {
                        "model_name": self.model_name,
                        "size": model_info.get("size", "unknown"),
                        "modified_at": model_info.get("modified_at", "unknown")
                    }
                return {"model_name": self.model_name, "status": "available"}
            
            # Fallback - just return the model name if we can't get detailed info
            return {"model_name": self.model_name, "status": "unknown"}
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"model_name": self.model_name, "status": "error", "error": str(e)}
    
    def generate_response(self, prompt: str, max_tokens=256, temperature=0.7, stream=False) -> str:
        """Generate a response from the LLM"""
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        try:
            response = requests.post(
                self.api_endpoint,
                headers={"Content-Type": "application/json"},
                data=json.dumps(data)
            )
            
            if response.status_code != 200:
                logger.error(f"Error from Ollama API: {response.status_code} - {response.text}")
                return f"Error generating response: {response.status_code}"
            
            response_json = response.json()
            generated_text = response_json.get('response', 'No response')
            
            return generated_text
        except Exception as e:
            logger.error(f"Exception during LLM generation: {e}")
            return f"Error generating response: {str(e)}"

class CharacterManager:
    """Manages NPC character profiles"""
    
    def __init__(self, characters_dir="characters"):
        """Initialize the character manager"""
        self.characters_dir = characters_dir
        self.characters = self._load_characters()
        
    def _load_characters(self) -> Dict[str, Dict[str, Any]]:
        """Load all character profiles from the characters directory"""
        characters = {}
        
        if not os.path.exists(self.characters_dir):
            os.makedirs(self.characters_dir)
            logger.warning(f"Characters directory {self.characters_dir} was created. No characters loaded.")
            return characters
            
        character_files = os.listdir(self.characters_dir)
        for file in character_files:
            if file.endswith('.json'):
                try:
                    with open(f'{self.characters_dir}/{file}', 'r') as f:
                        character_data = json.load(f)
                        character_id = character_data.get('character_id')
                        if character_id:
                            characters[character_id] = character_data
                            logger.info(f"Loaded character: {character_id}")
                        else:
                            logger.warning(f"Character file {file} has no character_id, skipping")
                except Exception as e:
                    logger.error(f"Error loading character from {file}: {e}")
        
        logger.info(f"Loaded {len(characters)} character profiles")
        return characters
    
    def get_character(self, character_id: str) -> Dict[str, Any]:
        """Get a character profile by ID"""
        character = self.characters.get(character_id)
        if not character:
            logger.warning(f"Character {character_id} not found")
        return character

class ConversationTester:
    """Test interactions between NPCs using LLM-powered dialogue"""
    NPCA_dialogue = []
    NPCB_dialogue = []
    
    def __init__(self, config_path: str = "config/conversation_test.json"):
        """Initialize the conversation tester with configuration"""
        # Initialize LLM service
        self.llm_service = self._init_llm_service()
        
        # Character manager for loading and accessing character profiles
        self.character_manager = CharacterManager()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize state storage
        self.conversations = {}
        self.conversation_history = {}
        self.test_scenarios = self.config.get("test_scenarios", [])
        self.max_turns = self.config.get("max_turns", 10)
        self.results = []
    
    def _init_llm_service(self) -> OllamaLLMService:
        """Initialize the LLM service based on environment settings"""
        model_name = os.environ.get("OLLAMA_MODEL", "deepseek-r1:7b")
        ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
        ollama_port = os.environ.get("OLLAMA_PORT", "11434")
        base_url = f"http://{ollama_host}:{ollama_port}"
        
        logger.info(f"Initializing Ollama service with model {model_name} at {base_url}")
        
        try:
            return OllamaLLMService(model_name=model_name, base_url=base_url)
        except Exception as e:
            logger.error(f"Error initializing Ollama service: {e}")
            raise RuntimeError("Could not initialize LLM service. Please check your configuration.")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
            return {
                "test_scenarios": [
                    {
                        "name": "default_scenario",
                        "description": "Two NPCs discussing a topic",
                        "npcs": ["tavernkeeper", "mysterious_stranger"],
                        "initial_prompt": "You encounter each other in the tavern. Start a conversation.",
                        "environment": {
                            "location": "tavern",
                            "time_of_day": "evening",
                            "objects": ["tables", "chairs", "fireplace"]
                        }
                    }
                ],
                "max_turns": 10,
                "analysis": {
                    "track_personality_consistency": True,
                    "track_memory_usage": True,
                    "track_speech_pattern_adherence": True
                }
            }
    
    def construct_character_prompt(self, character_id: str, context: str) -> str:
        """Construct a prompt for LLM based on character profile and context"""
        character = self.character_manager.get_character(character_id)
        if not character:
            return f"You are {character_id}. {context}"
        
        # Build a comprehensive character prompt
        prompt_parts = [
            f"You are role-playing as {character['name']}, a character in a medieval fantasy setting.",
            f"\nCORE TRAITS: {', '.join(character.get('core_traits', []))}",
            f"\nBACKGROUND: {character.get('background', '')}",
            f"\nSPEECH PATTERN: {character.get('speech_pattern', '')}"
        ]
        
        # Add knowledge boundaries if available
        if 'knowledge_boundaries' in character:
            prompt_parts.append("\nYOU KNOW ABOUT:")
            for knowledge in character['knowledge_boundaries']:
                prompt_parts.append(f"- {knowledge}")
        
        # Add relationships relevant to this conversation
        if 'relationships' in character:
            prompt_parts.append("\nYOUR RELATIONSHIPS:")
            for other_npc, relationship in character['relationships'].items():
                prompt_parts.append(f"- {other_npc}: {relationship}")
        
        # Add context and instruction
        prompt_parts.append(f"\n\nCURRENT SITUATION: {context}")
        prompt_parts.append("\nRespond in character with 1-3 sentences. Never break character. Never narrate actions. Speak as your character would speak.")
        
        return "\n".join(prompt_parts)
    
    def setup_conversation(self, npc_ids: List[str], scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Set up a conversation between NPCs within a scenario"""
        conversation_id = f"conv_{int(time.time())}"
        
        # Initialize conversation state
        self.conversations[conversation_id] = {
            "npcs": npc_ids,
            "scenario": scenario_data,
            "turns": 0,
            "messages": [],
            "environment": scenario_data.get("environment", {})
        }
        
        # Initialize history for each NPC in this conversation
        for npc_id in npc_ids:
            if npc_id not in self.conversation_history:
                self.conversation_history[npc_id] = {}
            
            self.conversation_history[npc_id][conversation_id] = deque(maxlen=5)  # Short-term memory
        
        # Save conversation to a text file
        output_dir = "conversations"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/conversation_{conversation_id}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"=== Conversation: {scenario_data['name']} ===\n")
            f.write(f"Setting: {scenario_data['environment'].get('location', 'Unknown')}, "
                f"{scenario_data['environment'].get('time_of_day', 'day')}\n")
            f.write(f"NPCs: {', '.join(npc_ids)}\n")
            f.write("\n=== Dialogue ===\n")
            
            for msg in self.conversations[conversation_id]["messages"]:
                f.write(f"\n[Turn {msg['turn']}] {msg['speaker']}: {msg['content']}\n")
            
            f.write("\n=== End of Conversation ===\n")
        
        logger.info(f"Conversation saved to {filename}")
        
        return self.conversations[conversation_id]
    
    def get_npc_response(self, npc_id: str, message: str, context: Dict[str, Any]) -> str:
        """Get a response from an NPC using the LLM"""
        # Construct context description
        context_desc = f"Location: {context.get('current_location', 'unknown')}. "
        context_desc += f"Time: {context.get('time_of_day', 'day')}. "
        
        if 'weather' in context:
            context_desc += f"Weather: {context['weather']}. "
        
        if 'objects' in context and context['objects']:
            context_desc += f"Nearby: {', '.join(context['objects'])}. "
        
        # Construct full prompt including character profile and context
        full_prompt = self.construct_character_prompt(npc_id, f"{context_desc} {message}")
        
        # Get response from LLM
        logger.debug(f"Prompt for {npc_id}: {full_prompt}")
        response = self.llm_service.generate_response(
            full_prompt,
            max_tokens=150,
            temperature=0.7
        )
        #print(f"Response from {npc_id}: {response}")
        # Clean up response if needed
        response, thinking = self._clean_response(response, npc_id)
        #print(f"Cleaned Response from {npc_id}: {response}")

        return response, thinking
    
    def _clean_response(self, response: str, npc_id: str) -> str:
        """Clean up LLM response to ensure it's in character"""
        # Remove common prefixes like "Character name:" or "I say:"
        removed_thoughts = re.findall(r"<think>.*?</think>", response, flags=re.DOTALL)
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        prefixes_to_remove = [
            f"{npc_id}:", 
            f"{npc_id.capitalize()}:",
            "I say:",
            "I respond:",
            "*thinking*",
            "*in character*"
        ]
        
        # # Also try to remove the character's name if available
        # character = self.character_manager.get_character(npc_id)
        # if character and 'name' in character:
        #     prefixes_to_remove.extend([
        #         f"{character['name']}:",
        #         f"{character['name'].split()[0]}:"  # First name only
        #     ])
        
        # Strip out prefixes
        response_text = response
        for prefix in prefixes_to_remove:
            if response_text.startswith(prefix):
                response_text = response_text[len(prefix):].strip()
        
        # # Remove surrounding quotes if present
        # if (response_text.startswith('"') and response_text.endswith('"')) or \
        #    (response_text.startswith("'") and response_text.endswith("'")):
        #     response_text = response_text[1:-1].strip()
            
        return response_text, removed_thoughts
    
    def run_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Run a complete conversation between NPCs"""
        conversation = self.conversations[conversation_id]
        npcs = conversation["npcs"]
        scenario = conversation["scenario"]
        
        logger.info(f"Starting conversation: {scenario['name']}")
        logger.info(f"NPCs involved: {', '.join(npcs)}")
        logger.info(f"Scenario: {scenario['description']}")
        
        # Add the initial prompt to start the conversation
        initial_prompt = scenario["initial_prompt"]
        
        # Create game state with environmental context
        game_state = {
            "current_location": conversation["environment"].get("location", "unknown"),
            "time_of_day": conversation["environment"].get("time_of_day", "day"),
            "weather": conversation["environment"].get("weather", "clear"),
            "npcs_present": npcs,
            "objects": conversation["environment"].get("objects", []),
            "conversation_history": [
            {"speaker": msg["speaker"], "content": msg["content"]} for msg in conversation["messages"]
            ]
        }
        
        # # Raise error if conversation history is empty
        # if not game_state["conversation_history"]:
        #     raise ValueError("Conversation history is empty. Cannot proceed with the conversation.")
        
        # Start with an initial prompt to the first NPC
        current_message = initial_prompt
        current_speaker = None
        
        # Main conversation loop
        for turn in range(self.max_turns):
            # Alternate between NPCs
            speaker_npc = npcs[turn % len(npcs)]
            listener_npc = npcs[(turn + 1) % len(npcs)]
            
            logger.info(f"Turn {turn+1}: {speaker_npc} responds to: '{current_message}'")
            
            # For the first turn, we use the initial prompt
            # For subsequent turns, we use the previous NPC's response
            if turn == 0:
                # For first turn, we need special handling - treating the initial prompt as context
                prompt = f"The scene: {scenario['description']}. {current_message} You are {speaker_npc}. How do you begin the conversation with {listener_npc}?"
                response, thinking = self.get_npc_response(speaker_npc, prompt, game_state)
            else:
                # For all other turns, handle as normal NPC-to-NPC dialogue
                # Construct a message that gives context of who is speaking to whom
                framed_message = f"{current_speaker} says to you: \"{current_message}\""
                response, thinking = self.get_npc_response(speaker_npc, framed_message, game_state)
            
            # Store the interaction in conversation history 
            self.conversation_history[speaker_npc][conversation_id].append({
                "turn": turn + 1,
                "message_received": current_message,
                "response": response,
                "speaking_to": listener_npc
            })
            
            # Add to conversation messages
            conversation["messages"].append({
                "turn": turn + 1,
                "speaker": speaker_npc,
                "listener": listener_npc,
                "thought" : thinking,
                "content": response,
                "timestamp": time.time()
            })
            
            logger.info(f"{speaker_npc} responds: '{response}'")
            
            # Set up for next turn
            current_message = response
            current_speaker = speaker_npc
            conversation["turns"] += 1
            
        logger.info(f"Conversation completed after {conversation['turns']} turns")
        return conversation["messages"]
    
    def analyze_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Analyze a completed conversation for quality and coherence"""
        conversation = self.conversations[conversation_id]
        messages = conversation["messages"]
        
        # Analysis metrics
        analysis = {
            "total_turns": conversation["turns"],
            "avg_response_length": sum(len(msg["content"]) for msg in messages) / len(messages),
            "topic_coherence": self._analyze_topic_coherence(messages),
            "personality_consistency": {},
            "memory_usage": {}
        }
        
        # For each NPC in the conversation, evaluate how well they maintained
        # their personality traits and used previous context
        for npc_id in conversation["npcs"]:
            character = self.character_manager.get_character(npc_id)
            npc_messages = [msg for msg in messages if msg["speaker"] == npc_id]
            
            if character and npc_messages:
                # Calculate personality consistency
                # This could be enhanced with an LLM-based analysis
                traits = character.get("core_traits", [])
                speech_pattern = character.get("speech_pattern", "")
                
                trait_score = self._analyze_trait_consistency(traits, npc_messages)
                speech_score = self._analyze_speech_consistency(speech_pattern, npc_messages)
                
                analysis["personality_consistency"][npc_id] = {
                    "trait_alignment": trait_score,
                    "speech_adherence": speech_score
                }
                
                # Check memory usage - how well did the NPC maintain context?
                npc_history = self.conversation_history[npc_id][conversation_id]
                memory_score = self._analyze_memory_consistency(npc_id, list(npc_history), conversation["scenario"])
                analysis["memory_usage"][npc_id] = memory_score
        
        return analysis
    
    def _analyze_trait_consistency(self, traits: List[str], messages: List[Dict[str, Any]]) -> float:
        """Analyze how consistently an NPC's messages align with their personality traits"""
        if not traits or not messages:
            return 0.5  # Neutral score with no data
            
        # Simple keyword matching approach
        # This could be enhanced with LLM-based analysis
        trait_words = []
        for trait in traits:
            # For each trait, add some related words
            if trait.lower() == "gruff":
                trait_words.extend(["direct", "blunt", "terse", "short", "grumble"])
            elif trait.lower() == "diplomatic":
                trait_words.extend(["careful", "thoughtful", "measured", "balanced", "fair"])
            elif trait.lower() == "wise":
                trait_words.extend(["insight", "experience", "knowledge", "patience", "understanding"])
            elif trait.lower() == "brave":
                trait_words.extend(["courage", "valor", "fearless", "bold", "daring"])
            else:
                # Default - just use the trait itself
                trait_words.append(trait.lower())
        
        # Check each message for trait words
        trait_matches = 0
        for msg in messages:
            content = msg["content"].lower()
            if any(word in content for word in trait_words):
                trait_matches += 1
        
        return trait_matches / len(messages) if messages else 0
    
    def _analyze_speech_consistency(self, speech_pattern: str, messages: List[Dict[str, Any]]) -> float:
        """Analyze how consistently an NPC's messages match their speech pattern"""
        if not speech_pattern or not messages:
            return 0.5  # Neutral score with no data
            
        # Extract key phrases from speech pattern
        pattern_parts = speech_pattern.lower().split('.')
        key_patterns = []
        
        for part in pattern_parts:
            if "short sentences" in part:
                key_patterns.append("short_sentences")
            if "proverbs" in part or "sayings" in part:
                key_patterns.append("uses_proverbs")
            if "dialect" in part:
                key_patterns.append("uses_dialect")
            if "formal" in part:
                key_patterns.append("formal_speech")
            if "metaphor" in part:
                key_patterns.append("uses_metaphors")
        
        # Check each message for adherence to patterns
        pattern_scores = []
        for msg in messages:
            content = msg["content"].lower()
            msg_score = 0
            
            # Check for short sentences
            if "short_sentences" in key_patterns:
                sentences = content.split('.')
                avg_words = sum(len(s.split()) for s in sentences if s.strip()) / max(1, len([s for s in sentences if s.strip()]))
                if avg_words < 10:
                    msg_score += 1
            
            # Check for proverbs/sayings (simple heuristic)
            if "uses_proverbs" in key_patterns:
                if any(phrase in content for phrase in ["they say", "as the saying goes", "old wisdom", "proverb"]):
                    msg_score += 1
            
            # Scale score to 0-1 range
            pattern_scores.append(msg_score / max(1, len(key_patterns)))
        
        return sum(pattern_scores) / len(messages) if messages else 0
        
    def _analyze_topic_coherence(self, messages: List[Dict[str, Any]]) -> float:
        """Analyze how well the conversation maintained coherent topics"""
        # Simplified implementation - could be enhanced with NLP
        # Here we're making a basic estimation based on message length consistency and overlap
        
        if len(messages) < 2:
            return 1.0  # Perfect score for a single message
            
        # Calculate variance in message length as one signal of coherence
        lengths = [len(msg["content"]) for msg in messages]
        avg_length = sum(lengths) / len(lengths)
        length_variance = sum((length - avg_length) ** 2 for length in lengths) / len(lengths)
        
        # Normalize to a 0-1 scale (lower variance = better coherence)
        # This is a very simplistic approach - real analysis would use NLP
        normalized_variance = min(1.0, 1.0 / (1 + length_variance / 1000))
        
        return normalized_variance
    
    def _analyze_memory_consistency(self, npc_id: str, history: List[Dict[str, Any]], 
                                   scenario: Dict[str, Any]) -> float:
        """Analyze how consistently an NPC references previous context"""
        # Simple implementation - could be enhanced with more sophisticated analysis
        
        if len(history) < 2:
            return 1.0  # Perfect score with minimal history
            
        # Check for simple references to previous messages
        # Count responses that reference something from a previous message
        reference_count = 0
        
        for i, entry in enumerate(history[1:], 1):  # Start from the second message
            current_msg = entry["response"].lower()
            prev_msgs = [h["message_received"].lower() for h in history[:i]]
            
            # Simple heuristic: look for any significant word overlap
            for prev_msg in prev_msgs:
                prev_words = set(prev_msg.split())
                significant_words = {w for w in prev_words if len(w) > 4}  # Only count substantive words
                
                if any(word in current_msg for word in significant_words):
                    reference_count += 1
                    break
        
        # Calculate what percentage of messages showed evidence of memory
        memory_score = reference_count / (len(history) - 1) if len(history) > 1 else 1.0
        return memory_score
    
    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all test scenarios and analyze the results"""
        results = []
        
        for scenario in self.test_scenarios:
            logger.info(f"Running test scenario: {scenario['name']}")
            
            # Set up the conversation
            npcs = scenario.get("npcs", [])
            if len(npcs) < 2:
                logger.warning(f"Scenario {scenario['name']} needs at least 2 NPCs. Skipping.")
                continue
                
            conversation = self.setup_conversation(npcs, scenario)
            conversation_id = list(self.conversations.keys())[-1]
            
            # Run the conversation
            messages = self.run_conversation(conversation_id)
            
            # Analyze the results
            analysis = self.analyze_conversation(conversation_id)
            
            # Store results
            results.append({
                "scenario": scenario["name"],
                "npcs": npcs,
                "messages": messages,
                "analysis": analysis
            })
            
            # Save the results to disk
            self.save_results(conversation_id, results[-1])
            
        return results
    
    def save_results(self, conversation_id: str, result: Dict[str, Any]) -> None:
        """Save the conversation and analysis results to disk"""
        output_dir = "test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{output_dir}/conversation_{conversation_id}.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
            
        logger.info(f"Results saved to {filename}")
    
    def print_conversation(self, conversation_id: str) -> None:
        """Print a conversation in a readable format"""
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            logger.error(f"Conversation {conversation_id} not found")
            return
            
        print(f"\n=== Conversation: {conversation['scenario']['name']} ===")
        print(f"Setting: {conversation['environment'].get('location', 'Unknown')}, "
              f"{conversation['environment'].get('time_of_day', 'day')}")
        print(f"NPCs: {', '.join(conversation['npcs'])}")
        print("\n=== Dialogue ===")
        
        for msg in conversation["messages"]:
            print(f"\n[Turn {msg['turn']}] {msg['speaker']}: {msg['content']}")
            
        print("\n=== End of Conversation ===\n")

def main():
    """Main entry point for the NPC conversation tester"""
    parser = argparse.ArgumentParser(description="Test conversations between NPCs using LLM")
    parser.add_argument("--config", type=str, default="config/npc-conversation-config.json",
                       help="Path to the configuration file")
    parser.add_argument("--scenario", type=str, help="Run a specific scenario by name")
    parser.add_argument("--turns", type=int, help="Override the number of conversation turns")
    parser.add_argument("--model", type=str, help="Override the Ollama model to use")
    args = parser.parse_args()
    
    # Set model from args if provided
    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model
    
    tester = ConversationTester(args.config)
    
    # Override max turns if specified
    if args.turns:
        tester.max_turns = args.turns
        
    # Run a specific scenario or all scenarios
    if args.scenario:
        # Find the specific scenario
        scenario = next((s for s in tester.test_scenarios if s["name"] == args.scenario), None)
        if scenario:
            logger.info(f"Running specific scenario: {args.scenario}")
            tester.test_scenarios = [scenario]
            results = tester.run_all_tests()
            
            # Print the conversation
            tester.print_conversation(list(tester.conversations.keys())[-1])
        else:
            logger.error(f"Scenario '{args.scenario}' not found in config")
            return
    else:
        # Run all scenarios
        results = tester.run_all_tests()
        
        # Print all conversations
        for conv_id in tester.conversations:
            tester.print_conversation(conv_id)
            
    
    # Print overall summary
    print("\n=== Test Summary ===")
    for result in results:
        scenario = result["scenario"]
        analysis = result["analysis"]
        
        print(f"\nScenario: {scenario}")
        print(f"NPCs: {', '.join(result['npcs'])}")
        print(f"Turns: {analysis['total_turns']}")
        print(f"Topic coherence: {analysis['topic_coherence']:.2f}")
        
        print("\nPersonality consistency:")
        for npc, scores in analysis['personality_consistency'].items():
            print(f"- {npc}: Trait {scores['trait_alignment']:.2f}, Speech {scores['speech_adherence']:.2f}")
            
        print("\nMemory usage:")
        for npc, score in analysis['memory_usage'].items():
            print(f"- {npc}: {score:.2f}")
            
    print("\n=== End of Test Summary ===")
    
    # Save conversation output to text file
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for conv_id, conversation in tester.conversations.items():
        filename = f"{output_dir}/conversation_{conv_id}.txt"
        with open(filename, 'w') as f:
            f.write(f"=== Conversation: {conversation['scenario']['name']} ===\n")
            f.write(f"Setting: {conversation['environment'].get('location', 'Unknown')}, "
                    f"{conversation['environment'].get('time_of_day', 'day')}\n")
            f.write(f"NPCs: {', '.join(conversation['npcs'])}\n")
            f.write("\n=== Dialogue ===\n")
            
            for msg in conversation["messages"]:
                f.write(f"\n[Turn {msg['turn']}] {msg['speaker']}: {msg['content']}\n")
                
            f.write("\n=== End of Conversation ===\n")
            
        logger.info(f"Conversation saved to {filename}")

if __name__ == "__main__":
    main()
