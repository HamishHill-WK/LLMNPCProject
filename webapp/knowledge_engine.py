import re
import time

class knowledge_engine:
    def __init__(self, character_id, knowledge_base):
        self.character_id = character_id
        self.knowledge_base = knowledge_base
        self.information_categories = [
            "personal_facts",        # Facts about the player character
            "world_knowledge",       # Information about the game world
            "opinions",              # Player's expressed opinions or preferences
            "relationships",         # Information about other characters
            "intentions",            # Player's stated goals or plans
            "background"             # Player's history or backstory
        ]
    
    def extract_information(self, player_message, conversation_context):
        """Extract potentially valuable information from player message"""
        extraction_prompt = f"""
        Analyze this message from the player: "{player_message}"
        
        Within the full conversation context:
        {conversation_context}
        
        Identify any NEW information the player has revealed about:
        1. Personal facts (name, occupation, traits, abilities)
        2. World knowledge (locations, events, facts about the game world)
        3. Opinions (likes, dislikes, beliefs)
        4. Relationships (connections to other characters)
        5. Intentions (goals, plans, desires)
        6. Background (history, backstory)
        
        For each piece of information identified:
        - State the exact information
        - Rate its confidence level (1-5)
        - Explain why this information seems to be new
        
        If no new information is detected, respond with "No new information detected."
        """
        
        extraction_results = self.llm_service.generate_response(extraction_prompt)
        return self._parse_extraction_results(extraction_results)
    
    def _parse_extraction_results(self, extraction_text):
        """Parse the extraction results into structured data"""
        if "No new information detected" in extraction_text:
            return []
        
        # Parse the extraction results into structured data format
        extracted_items = []
        
        # Split the text by numbered entries (1., 2., etc.)
        sections = re.split(r'\d+\.', extraction_text)[1:] if re.search(r'\d+\.', extraction_text) else []
        
        for section in sections:
            # Try to identify the category
            category = None
            for cat in self.information_categories:
                if cat.replace("_", " ") in section.lower():
                    category = cat
                    break
                
                # If no category found, try to determine from content
                if not category:
                    if any(term in section.lower() for term in ["name", "occupation", "trait", "ability", "skill"]):
                        category = "personal_facts"
                    elif any(term in section.lower() for term in ["location", "place", "world", "event"]):
                        category = "world_knowledge"
                    elif any(term in section.lower() for term in ["like", "dislike", "believe", "opinion", "feel"]):
                        category = "opinions"
                    elif any(term in section.lower() for term in ["friend", "enemy", "relationship", "connection"]):
                        category = "relationships"
                    elif any(term in section.lower() for term in ["goal", "plan", "desire", "intention", "want"]):
                        category = "intentions"
                    elif any(term in section.lower() for term in ["history", "background", "past", "backstory"]):
                        category = "background"
                    else:
                        category = "personal_facts"  # Default if we can't determine
                
                # Extract confidence level (1-5)
                confidence_match = re.search(r'confidence:?\s*(\d+)', section.lower())
                confidence = int(confidence_match.group(1)) if confidence_match else 3  # Default confidence
                
                # Extract the information and reasoning
                information = None
                reasoning = None
                
                lines = [line.strip() for line in section.split('\n') if line.strip()]
                if lines:
                    # First line or text before confidence is typically the information
                    information = lines[0].split("confidence")[0].strip() if "confidence" in lines[0].lower() else lines[0]
                
                # Look for reasoning after information
                reasoning_parts = [line for line in lines[1:] if "why" in line.lower() or "because" in line.lower()]
                reasoning = " ".join(reasoning_parts) if reasoning_parts else "Extracted from player message"
                
                if information:
                    extracted_items.append({
                        "category": category,
                        "information": information,
                        "confidence": confidence,
                        "reasoning": reasoning
                    })
                    
            return extracted_items
        
    def evaluate_importance(self, extracted_items, character_profile):
        """Determine which extracted information is worth remembering"""
        if not extracted_items:
            return []
        
        # Format the extracted items for evaluation
        items_text = "\n".join([f"- {item['category'].upper()}: {item['information']} (Confidence: {item['confidence']})" 
                               for item in extracted_items])
        
        evaluation_prompt = f"""
        As {self.character_id}, evaluate which of these pieces of information are worth remembering:
        
        {items_text}
        
        Consider:
        1. Relevance to your interests and goals: {', '.join(character_profile['goals'])}
        2. Potential future utility in conversations
        3. Alignment with your personality and what you would care about
        4. Confidence level in the information's accuracy
        
        For each item, decide:
        - REMEMBER: Definitely worth adding to long-term memory
        - MAYBE: Possibly worth remembering if it comes up again
        - IGNORE: Not relevant or useful to you
        
        Explain your reasoning for each decision.
        """
        
        evaluation_results = self.llm_service.generate_response(evaluation_prompt)
        return self._parse_evaluation_results(evaluation_results, extracted_items)
    
    def update_knowledge_base(self, important_items):
        """Add important information to the character's knowledge base"""
        for item in important_items:
            if item['decision'] == 'REMEMBER':
                # Add to knowledge base with metadata
                self.knowledge_base.add_knowledge(
                    category=item['category'],
                    information=item['information'],
                    source="player_statement",
                    confidence=item['confidence'],
                    timestamp=time.time(),
                    context=item.get('context', '')
                )
        
        # Return summary of updates
        return f"Added {len([i for i in important_items if i['decision'] == 'REMEMBER'])} new items to knowledge base"