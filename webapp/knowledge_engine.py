# player_knowledge_engine.py
# Knowledge extraction system for NPC memory management

import json
import os
import re
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PlayerKnowledgeEngine")

# Information categories that can be extracted about the player
KNOWLEDGE_CATEGORIES = [
    "identity",        # Name, race, appearance, etc.
    "background",      # Where they're from, their history
    "abilities",       # Skills, powers, combat style
    "possessions",     # Items, weapons, notable belongings
    "goals",           # What they want to achieve
    "relationships",   # Who they know, allies, enemies
    "preferences",     # What they like/dislike
    "knowledge"        # What information they possess
]

class KnowledgeEngine:
    """System for extracting, categorizing and storing knowledge about the player from conversations"""
    
    def __init__(self):
        """Initialize the knowledge engine"""
        self.knowledge_file = 'data/player_knowledge.json'
        self.knowledge_base = self._load_knowledge_base()
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load the knowledge base from disk or create a new one"""
        if os.path.exists(self.knowledge_file):
            try:
                with open(self.knowledge_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Error decoding knowledge base file. Creating new one.")
        
        # Initialize empty knowledge base with structure for each NPC
        return {
            "global": {category: [] for category in KNOWLEDGE_CATEGORIES},
            "npc_knowledge": {}
        }
    
    def _save_knowledge_base(self):
        """Save the knowledge base to disk"""
        with open(self.knowledge_file, 'w') as f:
            json.dump(self.knowledge_base, f, indent=2)
    
    def extract_player_knowledge(self, character_id: str, player_message: str, 
                                conversation_context: str, data_dict: dict, ollama_service) -> Dict[str, Any]:
        """
        Analyze player's message to extract knowledge about them
        
        Args:
            character_id: ID of the NPC who received the message
            player_message: The player's message
            conversation_context: Recent conversation history
            data_dict: Dictionary containing model information
            ollama_service: Service for generating LLM responses
            
        Returns:
            Dictionary with extraction results
        """
        # Save original prompt
        original_prompt = data_dict.get("prompt", "")
        
        # Create extraction prompt
        extraction_prompt = self._create_extraction_prompt(player_message, conversation_context)
        
        # Use LLM to extract knowledge
        data_dict["prompt"] = extraction_prompt
        
        try:
            # Get response from LLM
            extraction_result = ollama_service.get_response(data_dict, {}, None)
            
            # Restore original prompt
            data_dict["prompt"] = original_prompt
            
            # Parse extraction result
            extracted_items = self._parse_extraction_result(extraction_result)
            
            if not extracted_items:
                return {"new_knowledge": False, "items": []}
            
            # Update knowledge base
            self._update_knowledge_base(character_id, extracted_items)
            
            return {
                "new_knowledge": True,
                "items": extracted_items
            }
            
        except Exception as e:
            logger.error(f"Error extracting knowledge: {e}")
            # Restore original prompt
            data_dict["prompt"] = original_prompt
            return {"new_knowledge": False, "error": str(e)}
    
    def _create_extraction_prompt(self, player_message: str, conversation_context: str) -> str:
        """Create a prompt for extracting player knowledge"""
        categories_text = "\n".join([f"- {cat.title()}: Information about the player's {cat}" 
                                  for cat in KNOWLEDGE_CATEGORIES])
        
        return f"""<s>
You are analyzing a player's message in a role-playing game to extract any information the player reveals about their character.

The player's message is: "{player_message}"

Recent conversation context:
{conversation_context[:300] if len(conversation_context) > 300 else conversation_context}

Identify any explicit or strongly implied information about the player character in these categories:
{categories_text}

ONLY extract information that the player has clearly revealed about themselves. Do not make assumptions about information not directly stated or strongly implied.

For each piece of information you identify:
1. Specify which category it belongs to
2. State the exact information as a clear, concise statement
3. Rate your confidence (1-5 scale, where 1=uncertain, 5=certain)
4. Include the exact quote or strong implication from the player's message

Format your response as a JSON list where each item has these fields:
- "category": one of the categories listed above
- "information": the extracted information as a statement
- "confidence": numeric value 1-5
- "source": the specific part of the message that supports this

If no information about the player is revealed, respond with "NO_PLAYER_INFORMATION".
</s>"""
    
    def _parse_extraction_result(self, extraction_text: str) -> List[Dict[str, Any]]:
        """Parse the extraction result from the LLM"""
        # Check for no information response
        if "NO_PLAYER_INFORMATION" in extraction_text:
            return []
        
        try:
            # Try to find JSON in the response
            json_match = re.search(r'(\[.*?\])', extraction_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
                extracted_items = json.loads(json_text)
                
                # Validate and clean up items
                valid_items = []
                for item in extracted_items:
                    if "category" in item and "information" in item:
                        # Normalize category name
                        category = item["category"].lower()
                        if category not in KNOWLEDGE_CATEGORIES:
                            # Default to most appropriate category
                            if "name" in item["information"].lower():
                                category = "identity"
                            else:
                                category = "knowledge"
                        
                        # Ensure confidence is valid
                        if "confidence" not in item or not isinstance(item["confidence"], int):
                            item["confidence"] = 3  # Default medium confidence
                        else:
                            item["confidence"] = max(1, min(5, item["confidence"]))
                        
                        # Add normalized item
                        valid_items.append({
                            "category": category,
                            "information": item["information"],
                            "confidence": item["confidence"],
                            "source": item.get("source", "Implied from conversation"),
                            "timestamp": time.time()
                        })
                
                return valid_items
            
            # If no JSON found, try to parse text format
            return self._parse_text_extraction(extraction_text)
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from extraction result")
            return self._parse_text_extraction(extraction_text)
        
        except Exception as e:
            logger.error(f"Error parsing extraction result: {e}")
            return []
    
    def _parse_text_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Parse non-JSON extraction result"""
        items = []
        lines = text.split('\n')
        current_item = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for category indicators
            for category in KNOWLEDGE_CATEGORIES:
                category_title = category.title()
                if line.startswith(f"{category_title}:") or line.startswith(f"- {category_title}:"):
                    # Save previous item if exists
                    if current_item and "category" in current_item and "information" in current_item:
                        items.append(current_item)
                    
                    # Start new item
                    current_item = {"category": category, "confidence": 3, "timestamp": time.time()}
                    
                    # Extract information after the category
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        current_item["information"] = parts[1].strip()
                    break
            
            # Look for confidence indicators
            if "confidence" in line.lower() and ":" in line and current_item:
                try:
                    confidence_value = int(''.join(filter(str.isdigit, line.split(':', 1)[1])))
                    current_item["confidence"] = max(1, min(5, confidence_value))
                except ValueError:
                    pass
            
            # Look for source
            if "source" in line.lower() and ":" in line and current_item:
                current_item["source"] = line.split(':', 1)[1].strip()
        
        # Add the last item
        if current_item and "category" in current_item and "information" in current_item:
            items.append(current_item)
        
        return items
    
    def _update_knowledge_base(self, character_id: str, extracted_items: List[Dict[str, Any]]):
        """Update the knowledge base with new information"""
        # Initialize NPC's knowledge if not exists
        if "npc_knowledge" not in self.knowledge_base:
            self.knowledge_base["npc_knowledge"] = {}
        
        if character_id not in self.knowledge_base["npc_knowledge"]:
            self.knowledge_base["npc_knowledge"][character_id] = {
                category: [] for category in KNOWLEDGE_CATEGORIES
            }
        
        # Process each extracted item
        for item in extracted_items:
            category = item["category"]
            info = item["information"]
            confidence = item["confidence"]
            source = item.get("source", "Unknown")
            
            # Add to global knowledge (with higher confidence)
            self._add_to_global_knowledge(category, info, confidence, source)
            
            # Add to character's knowledge
            self._add_to_character_knowledge(character_id, category, info, confidence, source)
        
        # Save updated knowledge base
        self._save_knowledge_base()
    
    def _add_to_global_knowledge(self, category: str, info: str, confidence: int, source: str):
        """Add information to global knowledge"""
        if category not in self.knowledge_base["global"]:
            self.knowledge_base["global"][category] = []
        
        # Check if similar information exists
        is_new = True
        for existing in self.knowledge_base["global"][category]:
            if self._is_similar_information(existing["information"], info):
                # Update if new information has higher confidence
                if confidence > existing["confidence"]:
                    existing["information"] = info
                    existing["confidence"] = confidence
                    existing["last_updated"] = time.time()
                    existing["sources"].append(source)
                is_new = False
                break
        
        # Add new information
        if is_new:
            self.knowledge_base["global"][category].append({
                "information": info,
                "confidence": confidence,
                "first_learned": time.time(),
                "last_updated": time.time(),
                "sources": [source]
            })
    
    def _add_to_character_knowledge(self, character_id: str, category: str, info: str, confidence: int, source: str):
        """Add information to character's knowledge"""
        if category not in self.knowledge_base["npc_knowledge"][character_id]:
            self.knowledge_base["npc_knowledge"][character_id][category] = []
        
        # Check if similar information exists
        is_new = True
        for existing in self.knowledge_base["npc_knowledge"][character_id][category]:
            if self._is_similar_information(existing["information"], info):
                # Update if new information has higher confidence
                if confidence > existing["confidence"]:
                    existing["information"] = info
                    existing["confidence"] = confidence
                    existing["last_updated"] = time.time()
                is_new = False
                break
        
        # Add new information
        if is_new:
            self.knowledge_base["npc_knowledge"][character_id][category].append({
                "information": info,
                "confidence": confidence,
                "first_learned": time.time(),
                "last_updated": time.time()
            })
    
    def _is_similar_information(self, info1: str, info2: str) -> bool:
        """Check if two pieces of information are similar"""
        # Simple similarity check
        info1_norm = info1.lower().strip()
        info2_norm = info2.lower().strip()
        
        # Direct match
        if info1_norm == info2_norm:
            return True
        
        # One contains the other
        if info1_norm in info2_norm or info2_norm in info1_norm:
            return True
        
        return False
    
    def get_character_knowledge_about_player(self, character_id: str) -> Dict[str, List[str]]:
        """Get what a character knows about the player, formatted for prompts"""
        result = {}
        
        # Check if character has knowledge
        if "npc_knowledge" not in self.knowledge_base or \
           character_id not in self.knowledge_base["npc_knowledge"]:
            return result
        
        # Get character's knowledge
        for category, items in self.knowledge_base["npc_knowledge"][character_id].items():
            if not items:
                continue
            
            result[category] = []
            
            for item in items:
                # Format based on confidence
                knowledge_text = item["information"]
                if item["confidence"] < 3:
                    knowledge_text += " (not entirely certain)"
                
                result[category].append(knowledge_text)
        
        return result
    
    def format_player_knowledge(self, character_id: str) -> str:
        """Format player knowledge for inclusion in prompts"""
        knowledge = self.get_character_knowledge_about_player(character_id)
        
        if not knowledge:
            return "You don't know much about the player yet."
        
        # Format as sections by category
        sections = []
        
        for category, items in knowledge.items():
            if not items:
                continue
            
            category_title = category.title()
            section = f"{category_title}:\n"
            
            for item in items:
                section += f"- {item}\n"
            
            sections.append(section)
        
        return "\n".join(sections)


# Function to use in prompt_engine.py
def assess_player_knowledge(player_input: str, character_id: str, 
                           conversation_context: str, data_dict: dict, 
                           ollama_service) -> Dict[str, Any]:
    """
    Assess whether player input contains new knowledge
    
    Args:
        player_input: The player's message
        character_id: ID of the NPC receiving the message
        conversation_context: Recent conversation context
        data_dict: Dictionary containing model information
        ollama_service: Service for generating LLM responses
        
    Returns:
        Dictionary with assessment results
    """
    # Initialize knowledge engine if needed (singleton pattern)
    if not hasattr(assess_player_knowledge, "_engine"):
        assess_player_knowledge._engine = KnowledgeEngine()
    
    # Process the player input
    return assess_player_knowledge._engine.extract_player_knowledge(
        character_id=character_id,
        player_message=player_input,
        conversation_context=conversation_context,
        data_dict=data_dict,
        ollama_service=ollama_service
    )


def get_player_knowledge(character_id: str) -> str:
    """
    Get formatted knowledge about player for a character
    
    Args:
        character_id: ID of the NPC
        
    Returns:
        Formatted string of knowledge for inclusion in prompts
    """
    # Initialize knowledge engine if needed
    if not hasattr(assess_player_knowledge, "_engine"):
        assess_player_knowledge._engine = KnowledgeEngine()
    
    # Get formatted knowledge
    return assess_player_knowledge._engine.format_player_knowledge(character_id)