# knowledge_executive_planner.py
# Knowledge-enabled planner for analyzing and planning NPC responses

import json
import re
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KnowledgePlanner")

class DialogueContext:
    """Represents the current context of a dialogue interaction"""
    def __init__(self, character_id: str, player_message: str, game_state: Dict[str, Any], conversation_context: str = ""):
        self.character_id = character_id
        self.player_message = player_message
        self.game_state = game_state
        self.conversation_context = conversation_context
        self.analysis = {}
        self.knowledge_required = False
        self.knowledge_query = ""
        self.knowledge_result = None

class KnowledgeExecutivePlanner:
    """Analyzes player input and determines if external knowledge is required"""
    
    def __init__(self, ollama_service=None):
        self.ollama_service = ollama_service
        
        # Define common patterns for message classification
        self.patterns = {
            'greeting': r'\b(hello|hi|hey|greetings|good morning|good day|good evening|howdy)\b',
            'farewell': r'\b(goodbye|bye|farewell|see you|later|take care)\b',
            'question': r'\b(who|what|when|where|why|how|can you|could you|would you|do you|is there|are there)\b.*\?',
            'memory_recall': r'\b(remember|recall|earlier|before|last time|previously|you said|you told me|you mentioned)\b',
            'command': r'\b(go|take|give|show|tell|bring|find|get|put|use|open|close|attack|defend|move|run|walk|stop)\b',
            'emotional': r'\b(love|hate|angry|sad|happy|afraid|scared|worried|excited|proud|guilty|ashamed|disgusted)\b',
            # Knowledge-specific patterns
            'knowledge_query': r'\b(know|explain|tell me about|what|who|when|where|why|how|information on|details about|history of|meaning of)\b'
        }
        
        # Keywords that suggest knowledge might be needed
        self.knowledge_domains = [
            'history', 'science', 'geography', 'person', 'location', 'event', 
            'creature', 'item', 'spell', 'ability', 'rule', 'lore', 'legend',
            'craft', 'potion', 'weapon', 'armor', 'quest', 'mission'
        ]
    
    def analyze_message(self, context: DialogueContext) -> Dict[str, Any]:
        """Analyze player message to determine processing approach"""
        
        # Get player message and lowercase for pattern matching
        message = context.player_message
        
        # Perform basic pattern-based analysis first
        analysis = self._initial_pattern_analysis(message)
        
        # Analyze knowledge requirements
        knowledge_analysis = self._analyze_knowledge_needs(context)
        analysis.update(knowledge_analysis)

        # If LLM service is available, enhance analysis using it
        if self.ollama_service and len(message) > 10:  # Don't use LLM for very short messages
            llm_analysis = self._get_llm_analysis(context)
            # Merge LLM analysis with pattern analysis, prioritizing LLM
            analysis = {**analysis, **llm_analysis}
        
        # Store analysis in context
        context.analysis = analysis
        
        # Set knowledge required flag in context
        context.knowledge_required = analysis.get("knowledge_required", False)
        if context.knowledge_required:
            context.knowledge_query = analysis.get("knowledge_query", "")
            
        return analysis
    
    def _initial_pattern_analysis(self, message: str) -> Dict[str, Any]:
        """Perform initial pattern-based analysis of the message"""
        requires_memory = False
        memory_search_strategy = "none"
        memory_keywords = []
        
        message_types = []
        for type_name, pattern in self.patterns.items():
            if re.search(pattern, message, re.IGNORECASE):
                message_types.append(type_name)

        if "memory_recall" in message_types or "knowledge_query" in message_types or "question" in message_types:
            requires_memory = True
            memory_search_strategy = "semantic"
            # Extract potential keywords for memory search
            memory_keywords = [word for word in message.split() if len(word) > 2 and word not in ["remember", "recall", "said", "told", "mentioned", "the", "a", "yes" ] and word not in self.patterns['greetings'] and word not in self.patterns['farewell'] and word not in self.patterns['question']]
 
        analysis = {
            "message_types": message_types,
            "requires_memory": requires_memory,
            "memory_search_strategy": memory_search_strategy,
            "memory_search_keywords": memory_keywords
        } 
        return analysis
    
    def _analyze_knowledge_needs(self, context: DialogueContext) -> Dict[str, Any]:
        """Analyze if the message requires external knowledge to respond appropriately"""
        message = context.player_message.lower()
        
        # Check if this is a direct knowledge query
        is_knowledge_query = re.search(self.patterns['knowledge_query'], message) is not None
        
        # Check for domain-specific keywords that might indicate knowledge is needed
        knowledge_domains_found = []
        for domain in self.knowledge_domains:
            if domain.lower() in message:
                knowledge_domains_found.append(domain)
        
        # Check if NPC character would reasonably know this information
        # This requires checking against character knowledge boundaries
        character_id = context.character_id
        npc_knowledge_boundaries = self._get_npc_knowledge_boundaries(character_id, context.game_state)
        
        # Determine if this requires knowledge beyond NPC's defined knowledge
        # This is a simplistic approach - in a real system, you'd use more sophisticated
        # semantic analysis or LLM-based evaluation
        exceeds_npc_knowledge = is_knowledge_query and (
            len(knowledge_domains_found) > 0 or
            re.search(r'\b(what|who|when|where|why|how)\b.*\?', message)
        )
        
        if npc_knowledge_boundaries:
            for boundary in npc_knowledge_boundaries:
                boundary_lower = boundary.lower()
                # If any boundary specifically mentions this is within NPC knowledge
                for domain in knowledge_domains_found:
                    if domain.lower() in boundary_lower:
                        exceeds_npc_knowledge = False
                        break
        
        # Generate a knowledge query if needed
        knowledge_query = ""
        if exceeds_npc_knowledge:
            # Extract the core question or topic
            if "?" in message:
                knowledge_query = message
            else:
                # Extract key phrases that might be the subject of inquiry
                words = message.split()
                for domain in knowledge_domains_found:
                    pattern = rf'\b{domain}\s+of\s+(\w+|\w+\s+\w+)\b'
                    match = re.search(pattern, message)
                    if match:
                        knowledge_query = f"Information about {match.group(1)} {domain}"
                        break
                
                if not knowledge_query and len(words) > 3:
                    # Simple approach: take the latter part of the message as the query
                    knowledge_query = " ".join(words[len(words)//2:])
        
        # If this is clearly a knowledge query that exceeds the NPC's boundaries,
        # mark it as requiring external knowledge retrieval
        return {
            "knowledge_required": exceeds_npc_knowledge,
            "knowledge_domains": knowledge_domains_found,
            "knowledge_query": knowledge_query,
            "exceeds_npc_knowledge": exceeds_npc_knowledge
        }
    
    def _get_npc_knowledge_boundaries(self, character_id: str, game_state: Dict[str, Any]) -> List[str]:
        """Get the defined knowledge boundaries for an NPC"""
        # Check if the game state has character data
        if "all_characters" in game_state and character_id in game_state["all_characters"]:
            character_data = game_state["all_characters"].get(character_id, {})
            if "knowledge" in character_data:
                if isinstance(character_data["knowledge"], list):
                    return character_data["knowledge"]
                else:
                    return [character_data["knowledge"]]
            
            # Also check for knowledge_boundaries
            if "knowledge_boundaries" in character_data:
                return character_data["knowledge_boundaries"]
        
        # Return empty list if no knowledge boundaries found
        return []
    
    def _get_llm_analysis(self, context: DialogueContext) -> Dict[str, Any]:
        """Get enhanced analysis using the LLM"""

        prompt = f"""<system>
You are an AI system analyzing a player's message to an NPC in a game.
Provide a short analysis of the message to determine how to process it.

NPC: {context.character_id}
Player message: "{context.player_message}"

Analyze this message and provide a JSON response with the following:
1. message_type: One of [greeting, question, statement, request, command, emotional]
2. knowledge_required: Whether the NPC will need to access their deeper knowledge to answer this properly [true/false]
3. knowledge_query: If knowledge is required, what specific information needs to be retrieved
4. exceeds_npc_knowledge: Whether this question is outside what the NPC would know [true/false]

Respond with ONLY a JSON object and nothing else.
</system>
"""
        
        try:
            # Create data for ollama request
            data = {
                "model": "deepseek-r1:8b",
                "prompt": prompt,
                "stream": False,
                "max_tokens": 150,
                "temperature": 0.1  # Low temperature for more predictable analysis
            }
            
            # Create minimal game state for the request
            minimal_game_state = {
                "current_npc": context.character_id,
                "current_location": context.game_state.get("current_location", "unknown"),
                "all_characters": {}
            }
            
            # Get response from LLM
            llm_response = self.ollama_service.get_response(data, minimal_game_state, None)
            
            # Try to extract just the JSON part if there's any additional text
            json_match = re.search(r'({.*})', llm_response, re.DOTALL)
            if json_match:
                llm_response = json_match.group(1)
            
            # Parse JSON response
            llm_analysis = json.loads(llm_response)
            
            # Clean up booleans since JSON might have them as strings
            if "knowledge_required" in llm_analysis:
                if isinstance(llm_analysis["knowledge_required"], str):
                    llm_analysis["knowledge_required"] = llm_analysis["knowledge_required"].lower() == "true"
            
            if "exceeds_npc_knowledge" in llm_analysis:
                if isinstance(llm_analysis["exceeds_npc_knowledge"], str):
                    llm_analysis["exceeds_npc_knowledge"] = llm_analysis["exceeds_npc_knowledge"].lower() == "true"
            
            return llm_analysis
        except Exception as e:
            logger.error(f"Error getting LLM analysis: {e}")
            # Return empty dict to fall back to pattern analysis
            return {}
    
# Function for use in app.py
def analyze_for_knowledge(player_input: str, character_id: str, 
                        game_state: Dict[str, Any], conversation_context: str, 
                        ollama_manager=None) -> Dict[str, Any]:
    """
    Analyze a player message to determine if knowledge retrieval is needed
    
    Args:
        player_input: The player's message
        character_id: ID of the NPC receiving the message
        game_state: Current game state
        conversation_context: Recent conversation history
        ollama_manager: Service for LLM requests
        
    Returns:
        Dictionary with analysis results including knowledge requirements
    """
    # Initialize planner if needed (singleton pattern)
    if not hasattr(analyze_for_knowledge, "_planner"):
        analyze_for_knowledge._planner = KnowledgeExecutivePlanner(ollama_service=ollama_manager)
    
    # Create dialogue context
    context = DialogueContext(
        character_id=character_id,
        player_message=player_input,
        game_state=game_state,
        conversation_context=conversation_context
    )
    
    # Analyze message
    analyze_for_knowledge._planner.analyze_message(context)
    
    # Return analysis results
    return {
        "knowledge_required": context.knowledge_required,
        "knowledge_query": context.knowledge_query,
        "analysis": context.analysis
    }