# knowledge_engine.py
# Comprehensive knowledge extraction system for NPC memory management

import json
import os
import re
import logging
import time
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KnowledgeEngine")

# Information entity types
ENTITY_TYPES = [
    "player",        # Information about the player character
    "npc",           # Information about NPCs
    "location",      # Information about places
    "event",         # Information about events that happened
    "item",          # Information about objects, weapons, artifacts
    "faction",       # Information about groups, organizations
    "lore",          # General world knowledge and history
    "quest"          # Information about missions and objectives
]

json_format = """
{
    "entity_type": "",
    "category": "",
    "entity_name": "",
    "information": "",
    "confidence": 1-5,
    "source": ""
}
    """
# Information categories for each entity type
ENTITY_CATEGORIES = {
    "player": [
        "identity",      # Name, race, appearance, etc.
        "background",    # Where they're from, their history
        "abilities",     # Skills, powers, combat style
        "possessions",   # Items, weapons, notable belongings
        "goals",         # What they want to achieve
        "relationships", # Who they know, allies, enemies
        "preferences",   # What they like/dislike
        "knowledge"      # What information they possess
    ],
    "npc": [
        "identity",      # Name, title, appearance
        "location",      # Where they are/live
        "relationship",  # Relation to other characters
        "behavior",      # How they act, personality
        "knowledge",     # What they know
        "background",    # Their history, origin
        "abilities"      # Skills, powers they possess
    ],
    "location": [
        "description",   # What it looks like
        "inhabitants",   # Who/what lives there
        "dangers",       # Threats present there
        "resources",     # What can be found/harvested there
        "accessibility", # How to reach it, barriers
        "history",       # Past events that occurred there
        "atmosphere"     # The mood, weather, feel of the place
    ],
    "event": [
        "participants",  # Who was involved
        "outcome",       # What happened as a result
        "timeframe",     # When it happened
        "location",      # Where it happened
        "cause",         # Why it happened
        "significance"   # Why it matters
    ],
    "item": [
        "description",   # What it looks like
        "properties",    # Special abilities, features
        "origin",        # Where it came from
        "value",         # Worth or importance
        "ownership",     # Who possesses it
        "effects"        # What it does when used
    ],
    "faction": [
        "members",       # Who belongs to it
        "goals",         # What they want
        "territory",     # Where they operate
        "relationships", # Allies, enemies
        "resources",     # What they control
        "history",       # How they formed
        "reputation"     # How others view them
    ],
    "lore": [
        "historical",    # Past events
        "cultural",      # Customs, traditions
        "magical",       # Arcane knowledge
        "religious",     # Deities, beliefs
        "geographical",  # World features
        "political",     # Power structures
        "technological"  # Inventions, techniques
    ],
    "quest": [
        "objective",     # What needs to be done
        "giver",         # Who assigned it
        "rewards",       # What you get for completing it
        "location",      # Where it takes place
        "requirements",  # What's needed to complete it
        "obstacles",     # What makes it challenging
        "status"         # Current state of the quest
    ]
}

class KnowledgeEngine:
    """System for extracting, categorizing and storing knowledge from conversations"""
    
    def __init__(self):
        self.knowledge_file = 'data/game_knowledge.json'
        self.knowledge_base = self._load_knowledge_base()
        os.makedirs('data', exist_ok=True)
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        if os.path.exists(self.knowledge_file):
            try:
                with open(self.knowledge_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Error decoding knowledge base file. Creating new one.")
        
        # Initialize empty knowledge base
        knowledge_base = {
            "global": {},  # Shared knowledge across all NPCs
            "npc_knowledge": {},  # Knowledge specific to each NPC
        }
        
        # Initialize global categories for each entity type
        for entity_type in ENTITY_TYPES:
            knowledge_base["global"][entity_type] = {}
            for category in ENTITY_CATEGORIES[entity_type]:
                knowledge_base["global"][entity_type][category] = []
        
        return knowledge_base
    
    def _save_knowledge_base(self):
        """Save the knowledge base to disk"""
        print(f"Saving knowledge base to {self.knowledge_file}")
        with open(self.knowledge_file, 'w') as f:
            json.dump(self.knowledge_base, f, indent=2)
    
    def _create_extraction_prompt(self, player_message: str, conversation_context: str) -> str:
        """Create a prompt for extracting knowledge"""
        # Build a multi-level list of entity types and their categories
        entity_categories_text = ""
        for entity_type in ENTITY_TYPES:
            entity_categories_text += f"- {entity_type.upper()}:\n"
            for category in ENTITY_CATEGORIES[entity_type]:
                entity_categories_text += f"  - {category}: Information about {entity_type}'s {category}\n"
                        
        return f"""<system>
You are an ai system for analyzing a conversation to extract knowledge about various entities in a game world.

Your task is to identify any information about the following types of entities:
{entity_categories_text}

For each piece of information you identify:
1. Specify the entity type (player, npc, location, etc.)
2. Specify which category it belongs to within that entity type
3. Identify the specific entity being described (name of character, place, etc.)
4. State the exact information as a clear, concise statement
5. Rate your confidence (1-5 scale, where 1=uncertain, 5=certain)
6. Include the source quote or implication from the message

Format your response as a JSON list where each item has these fields:
- "entity_type": one of the entity types listed above
- "category": the appropriate category for this entity type
- "entity_name": the specific entity being described (e.g., "Bob", "Northern Kingdom", "Royal Guards")
- "information": the extracted information as a statement
- "confidence": numeric value 1-5
- "source": the specific part of the message that supports this

If no relevant information is revealed, respond with "NO_NEW_INFORMATION".
</system>

<text for analysis>
The player's most recent message is: "{player_message}"
</text for analysis>

<system>
In your response you MUST provide the information extracted from the player's message in the following format. Fill in the details for each extracted information item:
{json_format}
Any other format will be ignored by the system.
</system>
"""
    def _parse_extraction_result(self, extraction_text: str) -> List[Dict[str, Any]]:
        """Parse the extraction result from the LLM"""
        # Check for no information response
        if "NO_NEW_INFORMATION" in extraction_text:
            print(f"Knowledge Engine - No new information found")
            return []
        
        try:
            # Try to find JSON in the response
            print(f"Knowledge Engine - Parsing extraction result")
            # print(extraction_text)
            # print("\n")
            json_match = re.search(r'(\{.*?\}|\[.*?\])', extraction_text, re.DOTALL)
            if json_match:
                #print(f"Knowledge Engine {json_match.group(1)}")
                json_text = json_match.group(1)
                if json_text.startswith('{'):
                    json_text = f"[{json_text}]"
               # json_text = json_match.group(1)
                extracted_items = json.loads(json_text)
                
                # Validate and clean up items
                valid_items = []
                for item in extracted_items:
                    # Skip if missing required fields
                    if not all(key in item for key in ["entity_type", "category", "entity_name", "information"]):
                        #print(f"Knowledge Engine - Skipping item:\n {item}\n\n Failed to validate all required fields\n")
                        continue
                    
                    # Normalize entity type
                    entity_type = item["entity_type"].lower()
                    if entity_type not in ENTITY_TYPES:
                        # Try to find the closest match
                        if "player" in entity_type:
                            entity_type = "player"
                        elif "npc" in entity_type or "character" in entity_type:
                            entity_type = "npc"
                        elif "place" in entity_type or "area" in entity_type:
                            entity_type = "location"
                        else:
                            entity_type = "lore"  # Default to lore
                    
                    # Normalize category
                    category = item["category"].lower()
                    if category not in ENTITY_CATEGORIES.get(entity_type, []):
                        # Try to find the closest match or use default
                        if entity_type == "player" and "name" in category:
                            category = "identity"
                        elif entity_type == "npc" and "name" in category:
                            category = "identity"
                        elif entity_type == "location" and "look" in category:
                            category = "description"
                        else:
                            # Use the first category as default
                            category = ENTITY_CATEGORIES[entity_type][0]
                    
                    # Ensure confidence is valid
                    if "confidence" not in item or not isinstance(item["confidence"], int):
                        item["confidence"] = 3  # Default medium confidence
                    else:
                        item["confidence"] = max(1, min(5, item["confidence"]))
                    
                    # Add normalized item
                    valid_items.append({
                        "entity_type": entity_type,
                        "category": category,
                        "entity_name": item["entity_name"],
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
        
        # Patterns to look for
        entity_pattern = re.compile(r'(player|npc|location|event|item|faction|lore|quest)', re.IGNORECASE)
        category_pattern = re.compile(r'category:?\s*([a-z]+)', re.IGNORECASE)
        entity_name_pattern = re.compile(r'entity:?\s*([^:]+)', re.IGNORECASE)
        information_pattern = re.compile(r'information:?\s*([^:]+)', re.IGNORECASE)
        confidence_pattern = re.compile(r'confidence:?\s*(\d)', re.IGNORECASE)
        
        current_item = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for start of new item
            entity_match = entity_pattern.search(line)
            if entity_match and ("entity type" in line.lower() or line.lower().startswith(entity_match.group(1).lower())):
                # Save previous item if exists
                if current_item and "entity_type" in current_item and "information" in current_item:
                    items.append(current_item)
                
                # Start new item
                current_item = {
                    "entity_type": entity_match.group(1).lower(),
                    "confidence": 3,
                    "timestamp": time.time()
                }
                continue
                
            # Check for category
            category_match = category_pattern.search(line)
            if category_match and current_item:
                current_item["category"] = category_match.group(1).lower()
                
            # Check for entity name
            entity_name_match = entity_name_pattern.search(line)
            if entity_name_match and current_item:
                current_item["entity_name"] = entity_name_match.group(1).strip()
                
            # Check for information
            information_match = information_pattern.search(line)
            if information_match and current_item:
                current_item["information"] = information_match.group(1).strip()
                
            # Check for confidence
            confidence_match = confidence_pattern.search(line)
            if confidence_match and current_item:
                current_item["confidence"] = int(confidence_match.group(1))
                
            # Check for source
            if "source:" in line.lower() and current_item:
                current_item["source"] = line.split(":", 1)[1].strip()
        
        # Add the last item
        if current_item and "entity_type" in current_item and "entity_name" in current_item and "information" in current_item:
            items.append(current_item)
        
        return items
    
    def _update_knowledge_base(self, character_id: str, extracted_items: List[Dict[str, Any]]):
        """Update the knowledge base with new information"""
        # Initialize NPC's knowledge if not exists
        if "npc_knowledge" not in self.knowledge_base:
            self.knowledge_base["npc_knowledge"] = {}
        
        if character_id not in self.knowledge_base["npc_knowledge"]:
            self.knowledge_base["npc_knowledge"][character_id] = {}
            for entity_type in ENTITY_TYPES:
                self.knowledge_base["npc_knowledge"][character_id][entity_type] = {}
                for category in ENTITY_CATEGORIES[entity_type]:
                    self.knowledge_base["npc_knowledge"][character_id][entity_type][category] = []
        
        # Process each extracted item
        for item in extracted_items:
            entity_type = item["entity_type"]
            category = item["category"]
            entity_name = item["entity_name"]
            info = item["information"]
            confidence = item["confidence"]
            source = item.get("source", "Unknown")
            
            # Add to global knowledge
            self._add_to_global_knowledge(entity_type, category, entity_name, info, confidence, source)
            
            # Add to character's knowledge
            self._add_to_character_knowledge(character_id, entity_type, category, entity_name, info, confidence, source)
        
        # Save updated knowledge base
        self._save_knowledge_base()
    
    def _add_to_global_knowledge(self, entity_type: str, category: str, entity_name: str, 
                               info: str, confidence: int, source: str):
        """Add information to global knowledge"""
        # Ensure the entity type and category exist
        if entity_type not in self.knowledge_base["global"]:
            self.knowledge_base["global"][entity_type] = {}
        
        if category not in self.knowledge_base["global"][entity_type]:
            self.knowledge_base["global"][entity_type][category] = []
        
        # Check if similar information exists for this entity
        is_new = True
        for existing in self.knowledge_base["global"][entity_type][category]:
            if existing.get("entity_name") == entity_name and self._is_similar_information(existing["information"], info):
                # Update if new information has higher confidence
                if confidence > existing["confidence"]:
                    existing["information"] = info
                    existing["confidence"] = confidence
                    existing["last_updated"] = time.time()
                    if "sources" not in existing:
                        existing["sources"] = []
                    existing["sources"].append(source)
                is_new = False
                break
        
        # Add new information
        if is_new:
            self.knowledge_base["global"][entity_type][category].append({
                "entity_name": entity_name,
                "information": info,
                "confidence": confidence,
                "first_learned": time.time(),
                "last_updated": time.time(),
                "sources": [source]
            })
    
    def _add_to_character_knowledge(self, character_id: str, entity_type: str, category: str, 
                                   entity_name: str, info: str, confidence: int, source: str):
        """Add information to character's knowledge"""
        # Ensure the entity type and category exist
        if entity_type not in self.knowledge_base["npc_knowledge"][character_id]:
            self.knowledge_base["npc_knowledge"][character_id][entity_type] = {}
        
        if category not in self.knowledge_base["npc_knowledge"][character_id][entity_type]:
            self.knowledge_base["npc_knowledge"][character_id][entity_type][category] = []
        
        # Check if similar information exists
        is_new = True
        for existing in self.knowledge_base["npc_knowledge"][character_id][entity_type][category]:
            if existing.get("entity_name") == entity_name and self._is_similar_information(existing["information"], info):
                # Update if new information has higher confidence
                if confidence > existing["confidence"]:
                    existing["information"] = info
                    existing["confidence"] = confidence
                    existing["last_updated"] = time.time()
                is_new = False
                break
        
        # Add new information
        if is_new:
            self.knowledge_base["npc_knowledge"][character_id][entity_type][category].append({
                "entity_name": entity_name,
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
    
    def get_character_knowledge(self, character_id: str, entity_type: str = None, entity_name: str = None) -> Dict[str, Any]:
        """
        Get what a character knows, optionally filtered by entity type or specific entity
        
        Args:
            character_id: ID of the NPC
            entity_type: Optional filter for specific entity type
            entity_name: Optional filter for specific entity
            
        Returns:
            Dictionary of knowledge grouped by entity type and category
        """
        result = {}
        
        # Check if character has knowledge
        if "npc_knowledge" not in self.knowledge_base or \
            character_id not in self.knowledge_base["npc_knowledge"]:
            return result
        
        # Filter by entity type if specified
        entity_types = [entity_type] if entity_type else ENTITY_TYPES
        
        # Get character's knowledge
        for etype in entity_types:
            if etype not in self.knowledge_base["npc_knowledge"][character_id]:
                continue
                
            result[etype] = {}
            
            for category, items in self.knowledge_base["npc_knowledge"][character_id][etype].items():
                if not items:
                    continue
                
                result[etype][category] = []
                
                for item in items:
                    # Filter by entity name if specified
                    if entity_name and item.get("entity_name") != entity_name:
                        continue
                        
                    # Format based on confidence
                    knowledge_text = f"{item['entity_name']}: {item['information']}"
                    if item["confidence"] < 3:
                        knowledge_text += " (not entirely certain)"
                    
                    result[etype][category].append(knowledge_text)
        
        return result
            
    def format_entity_knowledge(self, character_id: str, entity_type: str, entity_name: str) -> str:
        """
        Format knowledge about a specific entity
        
        Args:
            character_id: ID of the NPC
            entity_type: Type of entity (location, npc, etc.)
            entity_name: Name of the specific entity
            
        Returns:
            Formatted string of knowledge about the entity
        """
        knowledge = self.get_character_knowledge(character_id, entity_type=entity_type)
        
        if not knowledge or entity_type not in knowledge:
            return f"You don't know much about {entity_name}."
        
        # Collect information about this specific entity
        entity_info = {}
        
        for category, items in knowledge[entity_type].items():
            entity_info[category] = []
            
            for item in items:
                if entity_name.lower() in item.lower():
                    # Extract just the information part after the entity name
                    parts = item.split(":", 1)
                    if len(parts) > 1:
                        entity_info[category].append(parts[1].strip())
        
        # Format as sections by category
        sections = []
        
        for category, items in entity_info.items():
            if not items:
                continue
            
            category_title = category.title()
            section = f"{category_title}:\n"
            
            for item in items:
                section += f"- {item}\n"
            
            sections.append(section)
        
        if not sections:
            return f"You don't know much about {entity_name}."
            
        return "\n".join(sections)
    
    def format_all_knowledge(self, character_id: str) -> str:
        """Format all knowledge a character has for debugging"""
        knowledge = self.get_character_knowledge(character_id)
        
        if not knowledge:
            return "This character doesn't have any knowledge yet."
        
        # Format as sections by entity type and category
        sections = []
        
        for entity_type, categories in knowledge.items():
            type_section = f"=== {entity_type.upper()} ===\n"
            has_content = False
            
            for category, items in categories.items():
                if not items:
                    continue
                
                category_section = f"{category.title()}:\n"
                
                for item in items:
                    category_section += f"- {item}\n"
                
                type_section += category_section + "\n"
                has_content = True
            
            if has_content:
                sections.append(type_section)
        
        if not sections:
            return "This character doesn't have any knowledge yet."
            
        return "\n".join(sections)

    # Function to use in prompt_engine.py
    def assess_knowledge(self, player_input: str, character_id: str, 
                        conversation_context: str, data_dict: dict, 
                        ollama_service) -> Dict[str, Any]:

        print("Extracting knowledge...")
        
        # Save original prompt
        original_prompt = data_dict.get("prompt", "")
        
        # Create extraction prompt
        extraction_prompt = self._create_extraction_prompt(player_input, conversation_context)
        
        
        # Save extraction prompt to a text file
        try:
            # Create extraction_prompts directory if it doesn't exist
            prompts_dir = 'data/extraction_prompts'
            os.makedirs(prompts_dir, exist_ok=True)
            
            # Create filename with timestamp and character ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prompts_dir}/prompt_{character_id}_{timestamp}.txt"
            
            # Write prompt to file
            with open(filename, 'w') as f:
                f.write(f"Character ID: {character_id}\n")
                f.write(f"Player message: {player_input}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write("-" * 50 + "\n")
                f.write(extraction_prompt)
            
            #logger.info(f"Saved extraction prompt to {filename}")
        except Exception as e:
            logger.error(f"Failed to save extraction prompt: {e}")
        # Use LLM to extract knowledge
        data_dict["prompt"] = extraction_prompt
        
        # Debug: Save data_dict to file for inspection
        try:
            # Create debug directory if it doesn't exist
            debug_dir = 'data/debug'
            os.makedirs(debug_dir, exist_ok=True)
            
            # Create filename with timestamp and character ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{debug_dir}/data_dict_{character_id}_{timestamp}.json"
            
            # Write data_dict to file
            with open(filename, 'w') as f:
                json.dump(data_dict, f, indent=2, default=str)
            
            logger.info(f"Saved data_dict to {filename}")
        except Exception as e:
            logger.error(f"Failed to save data_dict: {e}")
        minimal_game_state = {
            "current_npc": character_id,
            "current_location": "unknown",
            "all_characters": {}
        }
        try:
            # Get response from LLM
            extraction_result = ollama_service.get_response(data_dict, minimal_game_state, None)
            
            # Restore original prompt
            data_dict["prompt"] = original_prompt
            
            # Save extraction result to a text file
            try:
                # Create extraction_results directory if it doesn't exist
                extraction_dir = 'data/extraction_results'
                os.makedirs(extraction_dir, exist_ok=True)
                
                # Create filename with timestamp and character ID
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{extraction_dir}/extraction_{character_id}_{timestamp}.txt"
                
                # Write result to file
                with open(filename, 'w') as f:
                    f.write(f"Character ID: {character_id}\n")
                    f.write(f"Player message: {player_input}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write("-" * 50 + "\n")
                    f.write(extraction_result)
                
                logger.info(f"Saved extraction result to {filename}")
            except Exception as e:
                logger.error(f"Failed to save extraction result: {e}")
            
            # Parse extraction result
            extracted_items = self._parse_extraction_result(extraction_result)
            
            # Save parsed extracted items to file
            try:
                # Create extraction_results directory if it doesn't exist
                extraction_dir = 'data/extraction_results'
                os.makedirs(extraction_dir, exist_ok=True)
                
                # Create filename with timestamp and character ID
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{extraction_dir}/items_{character_id}_{timestamp}.json"
                
                # Write extracted items to file
                with open(filename, 'w') as f:
                    json.dump(extracted_items, f, indent=2)
                
                logger.info(f"Saved {len(extracted_items)} extracted items to {filename}")
            except Exception as e:
                logger.error(f"Failed to save extracted items: {e}")
            
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

    def get_player_knowledge(self, character_id: str) -> str:
        """Format knowledge about the player for inclusion in prompts"""
        knowledge = self.get_character_knowledge(character_id, entity_type="player")
        
        if not knowledge or "player" not in knowledge:
            return "You don't know much about the player yet."
        
        # Format as sections by category
        sections = []
        
        for category, items in knowledge["player"].items():
            if not items:
                continue
            
            category_title = category.title()
            section = f"{category_title}:\n"
            
            for item in items:
                section += f"- {item}\n"
            
            sections.append(section)
        
        if not sections:
            return "You don't know much about the player yet."
            
        return "\n".join(sections)

    def get_entity_knowledge(self, character_id: str, entity_type: str, entity_name: str) -> str:
        # Get formatted knowledge
        return self.format_entity_knowledge(character_id, entity_type, entity_name)