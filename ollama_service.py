# ollama_service.py - Connection to locally-hosted LLM using Ollama
import requests
import json
import time
import os
from threading import Thread, Event

class OllamaLLMService:
    def __init__(self, model_name="llama2", base_url="http://localhost:11434"):
        """
        Initialize the Ollama LLM service.
        
        Args:
            model_name (str): The model to use with Ollama (e.g., "llama2", "mistral", "phi")
            base_url (str): The URL where Ollama is running
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        
        # Check if Ollama is available and the model is loaded
        self._verify_service()
    
    def _verify_service(self):
        """Verify Ollama is running and model is available"""
        try:
            # Check if Ollama server is running
            response = requests.get(f"{self.api_url}/tags")
            if response.status_code != 200:
                raise ConnectionError(f"Could not connect to Ollama API at {self.api_url}")
            
            # Check if the specified model is available
            available_models = response.json().get("models", [])
            model_exists = any(model["name"] == self.model_name for model in available_models)
            
            if not model_exists:
                print(f"Model '{self.model_name}' not found in Ollama. Pulling it now...")
                self._pull_model()
            else:
                print(f"Model '{self.model_name}' is available and ready to use.")
                
        except requests.RequestException as e:
            raise ConnectionError(f"Error connecting to Ollama: {e}. Is Ollama running?")
    
    def _pull_model(self):
        """Pull the model from Ollama repository if not available locally"""
        try:
            # Send request to pull the model
            response = requests.post(
                f"{self.api_url}/pull",
                json={"name": self.model_name},
                stream=True  # Stream the response to monitor progress
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Failed to pull model: {response.text}")
            
            # Display progress
            print(f"Pulling model '{self.model_name}'...")
            for line in response.iter_lines():
                if line:
                    progress = json.loads(line.decode('utf-8'))
                    if 'status' in progress:
                        status = progress.get('status')
                        if status == 'success':
                            print(f"Successfully pulled model '{self.model_name}'")
                            return
                        print(f"Progress: {status}")
            
            print(f"Model '{self.model_name}' is now available")
            
        except Exception as e:
            raise RuntimeError(f"Error pulling model: {e}")
    
    def generate_response(self, prompt, max_new_tokens=150, temperature=0.7, stream=False):
        """
        Generate a response from the Ollama model.
        
        Args:
            prompt (str): The input prompt
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature (0.0 to 1.0)
            stream (bool): Whether to stream the response
            
        Returns:
            str: The generated response
        """
        # Prepare the request data
        request_data = {
            "model": self.model_name,
            "prompt": self._format_prompt_for_model(prompt),
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_new_tokens,
            }
        }
        
        try:
            if stream:
                return self._stream_response(request_data)
            else:
                return self._generate_response(request_data)
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I cannot respond right now."
    
    def _generate_response(self, request_data):
        """Generate a complete response at once"""
        response = requests.post(
            f"{self.api_url}/generate",
            json=request_data
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Error from Ollama API: {response.text}")
        
        result = response.json()
        return result.get("response", "").strip()
    
    def _stream_response(self, request_data):
        """Stream the response for progressive display"""
        # Create a background thread to collect the streamed response
        stop_event = Event()
        collected_response = [""]
        
        def collect_stream():
            response = requests.post(
                f"{self.api_url}/generate",
                json=request_data,
                stream=True
            )
            
            if response.status_code != 200:
                collected_response[0] = f"Error from Ollama API: {response.text}"
                stop_event.set()
                return
            
            full_text = ""
            for line in response.iter_lines():
                if stop_event.is_set():
                    break
                    
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    text_chunk = chunk.get("response", "")
                    full_text += text_chunk
                    
                    # Display progress
                    print(text_chunk, end="", flush=True)
                    
                    # Check if this is the last chunk
                    if chunk.get("done", False):
                        break
            
            collected_response[0] = full_text
            stop_event.set()
        
        # Start collecting in background
        thread = Thread(target=collect_stream)
        thread.start()
        
        # Wait for completion or timeout
        timeout = 60  # seconds
        thread.join(timeout)
        
        if not stop_event.is_set():
            stop_event.set()  # Signal to stop if timeout
            return "Response generation timed out."
        
        return collected_response[0].strip()
    
    def _format_prompt_for_model(self, prompt):
        """Format the prompt based on the model type"""
        # Format for Llama-2-Chat models
        if "llama2" in self.model_name.lower():
            return f"<s>[INST] {prompt} [/INST]"
        
        # Format for Mistral Instruct models
        elif "mistral" in self.model_name.lower():
            return f"<s>[INST] {prompt} [/INST]"
        
        # Format for Phi models
        elif "phi" in self.model_name.lower():
            return f"Instruct: {prompt}\nOutput:"
        
        # Default format for general instruction models
        return f"### Instruction:\n{prompt}\n\n### Response:"
    
    def get_model_info(self):
        """Return information about the currently loaded model"""
        try:
            response = requests.post(
                f"{self.api_url}/show",
                json={"name": self.model_name}
            )
            
            if response.status_code != 200:
                return {"model_name": self.model_name, "error": response.text}
            
            model_info = response.json()
            return {
                "model_name": self.model_name,
                "parameters": model_info.get("parameters"),
                "size": model_info.get("size"),
                "quantization_level": model_info.get("details", {}).get("quantization_level", "unknown"),
                "backend": "Ollama"
            }
        except Exception as e:
            return {"model_name": self.model_name, "error": str(e)}
    
    def list_available_models(self):
        """List all available models in Ollama"""
        try:
            response = requests.get(f"{self.api_url}/tags")
            if response.status_code != 200:
                return []
            
            return [model["name"] for model in response.json().get("models", [])]
        except Exception:
            return []

# Example usage
if __name__ == "__main__":
    # Simple test function
    def test_ollama_service():
        try:
            ollama = OllamaLLMService("llama2")
            
            # Get model info
            model_info = ollama.get_model_info()
            print(f"Model info: {json.dumps(model_info, indent=2)}")
            
            # Test non-streaming generation
            prompt = "Write a short poem about artificial intelligence."
            print(f"\nPrompt: {prompt}")
            print("\nGenerating response...")
            response = ollama.generate_response(prompt, max_new_tokens=100, temperature=0.7)
            print(f"Response: {response}")
            
            # Test streaming
            prompt = "Explain why local LLMs are useful."
            print(f"\nPrompt: {prompt}")
            print("\nGenerating streaming response...")
            response = ollama.generate_response(prompt, max_new_tokens=150, temperature=0.7, stream=True)
            print(f"\nFinal response: {response}")
            
        except Exception as e:
            print(f"Error testing Ollama service: {e}")
    
    test_ollama_service()