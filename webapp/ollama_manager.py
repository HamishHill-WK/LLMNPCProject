import json
import os
import requests
import datetime
import prompt_engine 
import re 

class OllamaManager:
    def __init__(self, prompt_engine):
        self.prompt_engine = prompt_engine
        
    def get_response(self, data, game_state, Mem_manager):
        # Save prompt to a text file
        if Mem_manager:
            data["prompt"] = self.prompt_engine.add_system_prompt(data, game_state, Mem_manager)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"prompt_{timestamp}.txt"
        os.makedirs("prompts", exist_ok=True)
        with open(f"prompts/{filename}", "w", encoding="utf-8") as f:
            f.write(data["prompt"])
            
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps(data)
        )
        
        # Ensure the response is JSON serializable
        response_json = response.json()
        return response_json.get('response', 'No response')

    def clean_response(character_response):
            # Extract the chain of thought (content between <think> tags)
        chain_of_thought_match = re.search(r'<think>(.*?)</think>', character_response, re.DOTALL)
        chain_of_thought = chain_of_thought_match.group(1).strip() if chain_of_thought_match else ""

        # Remove the <think> content from the character response
        clean_response = re.sub(r'<think>.*?</think>', '', character_response, flags=re.DOTALL).strip()
        
        #clean_response = clean_response.replace('<character_response>', '').replace('</character_response>', '').replace(f"{self.characters[character_id]['name']}: ", "").strip()

        return clean_response, chain_of_thought