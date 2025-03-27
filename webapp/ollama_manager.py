import json
import os
import requests
import datetime
import re 

class OllamaManager:
    def __init__(self, prompt_engine):
        self.prompt_engine = prompt_engine
        self.available_models = self.get_available_models()
        self.selected_model = self.available_models[0] if self.available_models else None
        
    def get_response(self, data, game_state, Mem_manager):
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

    def clean_response(self, character_response):
            # Extract the chain of thought (content between <think> tags)
        chain_of_thought_match = re.search(r'<think>(.*?)</think>', character_response, re.DOTALL)
        chain_of_thought = chain_of_thought_match.group(1).strip() if chain_of_thought_match else ""

        # Remove the <think> content from the character response
        clean_response = re.sub(r'<think>.*?</think>', '', character_response, flags=re.DOTALL).strip()
        
        return clean_response, chain_of_thought
    
    def get_available_models(self):
        """
        Get the list of available models from Ollama.
        
        Returns:
            list: A list of available model names.
        """
        try:
            response = requests.get("http://localhost:11434/api/tags")
            response.raise_for_status()  # Raises HTTPError for bad responses
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {str(e)}")
            return []