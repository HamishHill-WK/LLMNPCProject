# knowledge_executive_planner.py
# Knowledge-enabled planner for analyzing and planning NPC responses

import json
import re
import logging
from typing import Dict, Any, List 

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
    def __init__(self, ollama_service=None, knowledge_engine=None):
        self.ollama_service = ollama_service
        self.knowledge_engine = knowledge_engine
        
        #print(f"Knowledge Executive Planner initialized type {type(self.ollama_service)}" )
        
        # Define common patterns for message classification
        self.patterns = {
            'greeting': r'\b(hello|hi|hey|greetings|good morning|good day|good evening|howdy)\b',
            'farewell': r'\b(goodbye|bye|farewell|see you|later|take care)\b',
            'question': r'\b(who|what|when|where|why|how|can you|could you|would you|do you|is there|are there|was there)\b.*\?',
            'memory_recall': r'\b(remember|recall|earlier|before|last time|previously|you said|you told me|you mentioned)\b',
            'command': r'\b(go|take|give|show|tell|bring|find|get|put|use|open|close|attack|defend|move|run|walk|stop)\b',
            'emotional': r'\b(love|hate|angry|sad|happy|afraid|scared|worried|excited|proud|guilty|ashamed|disgusted)\b',
            'knowledge_query': r'\b(know|explain|tell me about|information on|details about|history of|meaning of)\b'
        }
        
        # Keywords that suggest knowledge might be needed
        self.knowledge_domains = [
            'history', 'science', 'geography', 'person', 'location', 'event', 
            'creature', 'item', 'spell', 'ability', 'rule', 'lore', 'legend',
            'craft', 'potion', 'weapon', 'armor', 'quest', 'mission'
        ]
    
    def set_ollama(self, ollama_service):
        self.ollama_service = ollama_service
    
    def _initial_pattern_analysis(self, message: str) -> Dict[str, Any]:
        """Perform initial pattern-based analysis of the message"""
        print("initial pattern analysis")
        requires_memory = False
        memory_search_strategy = "none"
        memory_keywords = []
        message_types = []
        
        if '?' in message:  # Check for question marks
            requires_memory = True
            message_types.append("question")
            memory_search_strategy = "semantic"
            memory_keywords = self.get_keywords(message)

        if requires_memory is False:
            for type_name, pattern in self.patterns.items():
                if re.search(pattern, message, re.IGNORECASE):
                    message_types.append(type_name)

            if "memory_recall" in message_types or "knowledge_query" in message_types or "question" in message_types:
                print("memory recall")
                requires_memory = True
                memory_search_strategy = "semantic"
                # Extract potential keywords for memory search
                memory_keywords = self.get_keywords(message)
    
        analysis = {
            "message_types": message_types,
            "requires_memory": requires_memory,
            "memory_search_strategy": memory_search_strategy,
            "memory_search_keywords": memory_keywords
        } 
        return analysis
    
    def get_keywords(self, message: str) -> List[str]:
        print("get keywords")
        return [word for word in message.split() if len(word) > 2 and word not in ["remember", "recall", "said", "told", "mentioned", "the", "a", "yes" ] and word not in self.patterns['greeting'] and word not in self.patterns['farewell'] and word not in self.patterns['question']]

    def _get_llm_analysis(self, context: DialogueContext) -> Dict[str, Any]:
        print("get llm analysis")
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
4. memory_search_strategy: How should the NPC search for relevant memories [semantic, keyword, none]
5. memory_search_keywords: Keywords or phrases that could be used to search for relevant memories
6. new_knowledge: Does the text contain information which might be useful to the npc later? [true/false]
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
            
            return llm_analysis
        except Exception as e:
            logger.error(f"Error getting LLM analysis: {e}")
            # Return empty dict to fall back to pattern analysis
            return {}
        
    # Function for use in app.py
    def analyze_for_knowledge(self, player_input: str, character_id: str, 
                            game_state: Dict[str, Any], conversation_context: str) -> Dict[str, Any]:
        context = DialogueContext(
            character_id=character_id,
            player_message=player_input,
            game_state=game_state,
            conversation_context=conversation_context
        )

        # Perform basic pattern-based analysis first
        analysis = self._initial_pattern_analysis(player_input)
        
        # Analyze knowledge requirements
        if analysis.get("requires_memory", True):
            print("knowledge required")
            llm_analysis = self._get_llm_analysis(context)
            # Merge LLM analysis with pattern analysis, prioritizing LLM
            analysis = {**analysis, **llm_analysis}
        
        return analysis