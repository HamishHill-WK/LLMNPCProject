import requests
import json
from typing import Dict, Any

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
        except Exception as e:
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
                return f"Error generating response: {response.status_code}"
            
            response_json = response.json()
            generated_text = response_json.get('response', 'No response')
            
            return generated_text
        except Exception as e:
            return f"Error generating response: {str(e)}"