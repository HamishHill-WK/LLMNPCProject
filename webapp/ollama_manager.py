import json
import os
import requests
import datetime
import re
import logging
import openai
import socket
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OllamaManager")

class OllamaManager:
    def __init__(self, prompt_engine):
        self.prompt_engine = prompt_engine
        
        # Get configuration from environment variables
        self.ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
        self.ollama_port = os.environ.get("OLLAMA_PORT", "11434")
        self.default_ollama_model = os.environ.get("DEFAULT_OLLAMA_MODEL", "deepseek-r1:8b")
        self.default_openai_model = os.environ.get("DEFAULT_OPENAI_MODEL", "gpt-3.5-turbo")
        self.debug_prompts = os.environ.get("DEBUG_PROMPTS", "true").lower() == "true"
        
        # Check if ollama hostname resolves, otherwise fallback to localhost
        self.check_ollama_connection()
        
        # AI provider settings
        self.ai_provider = os.environ.get("AI_PROVIDER", "ollama")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.openai_client = None
        if self.openai_api_key:
            try:
                self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
        # Get available models
        self.available_models = {
            "ollama": self.get_ollama_models(),
            "openai": self.get_openai_models()
        }
        
        # Set default model
        self.selected_model = (
            self.available_models["ollama"][0] 
            if self.available_models["ollama"] 
            else self.default_ollama_model
        )
        
        # Log configuration
        logger.info(f"OllamaManager initialized with: Ollama host={self.ollama_host}:{self.ollama_port}, Provider={self.ai_provider}")

    def check_ollama_connection(self):
        """Check if ollama hostname resolves, otherwise fallback to localhost"""
        try:
            # First try the configured host
            requests.get(f"http://{self.ollama_host}:{self.ollama_port}/api/version", timeout=1)
            logger.info(f"Successfully connected to Ollama at {self.ollama_host}:{self.ollama_port}")
        except requests.exceptions.ConnectionError:
            # If that fails, try localhost
            try:
                if self.ollama_host != "localhost":
                    logger.info(f"Could not connect to {self.ollama_host}, trying localhost")
                    requests.get(f"http://localhost:{self.ollama_port}/api/version", timeout=1)
                    logger.info(f"Successfully connected to Ollama at localhost:{self.ollama_port}")
                    self.ollama_host = "localhost"
                else:
                    logger.warning(f"Could not connect to Ollama at {self.ollama_host}:{self.ollama_port}")
            except requests.exceptions.ConnectionError:
                # If localhost doesn't work either, we'll keep the original setting
                # but warn the user
                logger.warning(f"Could not connect to Ollama at localhost:{self.ollama_port}")
                logger.warning("Please ensure Ollama is running")
        except Exception as e:
            logger.error(f"Error checking Ollama connection: {str(e)}")

    def set_ai_provider(self, provider, model=None, api_key=None):
        """Set the AI provider and optionally the model and API key"""
        if provider not in ["ollama", "openai"]:
            raise ValueError(f"Unsupported AI provider: {provider}")
        
        self.ai_provider = provider
        
        # Update OpenAI API key if provided
        if api_key and provider == "openai":
            self.openai_api_key = api_key
            try:
                self.openai_client = openai.OpenAI(api_key=api_key)
                # Refresh OpenAI models
                self.available_models["openai"] = self.get_openai_models()
                logger.info("OpenAI client initialized with new API key")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client with new API key: {e}")
                return False
        
        # Set selected model if provided
        if model:
            self.selected_model = model
        # Otherwise use the first available model for the selected provider
        elif self.available_models[provider]:
            self.selected_model = self.available_models[provider][0]
        
        return True
        
    def get_response(self, data, game_state, Mem_manager):
        """Get response from the selected AI provider"""
        if Mem_manager:
            data["prompt"] = self.prompt_engine.add_system_prompt(data, game_state, Mem_manager)

        # Save prompt for debugging
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"prompt_{timestamp}.txt"
        os.makedirs("prompts", exist_ok=True)
        with open(f"prompts/{filename}", "w", encoding="utf-8") as f:
            f.write(data["prompt"])
        
        # Use the appropriate method based on the selected provider
        if self.ai_provider == "openai":
            return self._get_openai_response(data)
        else:
            return self._get_ollama_response(data)
    
    def _get_ollama_response(self, data):
        """Get response from Ollama API"""
        try:
            response = requests.post(
                f"http://{self.ollama_host}:{self.ollama_port}/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(data)
            )
            
            # Ensure the response is JSON serializable
            response_json = response.json()
            return response_json.get('response', 'No response')
        except Exception as e:
            error_msg = f"Error getting response from Ollama: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _get_openai_response(self, data):
        """Get response from OpenAI API"""
        if not self.openai_client:
            return "OpenAI API key not set or invalid. Please configure your API key."
        
        try:
            # Extract parameters
            model = data.get("model", "gpt-3.5-turbo")
            max_tokens = data.get("max_tokens", 150)
            temperature = data.get("temperature", 0.7)
            prompt = data.get("prompt", "")
            
            # Format the messages for OpenAI
            # Look for system instruction in the prompt
            system_instruction = ""
            user_content = prompt
            
            # Extract system message if enclosed in <system> tags
            system_match = re.search(r'<system>(.*?)</system>', prompt, re.DOTALL)
            if system_match:
                system_instruction = system_match.group(1).strip()
                # Remove the system part from the user content
                user_content = re.sub(r'<system>.*?</system>', '', prompt, flags=re.DOTALL).strip()
            
            # Create messages array for chat completion
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            
            # Add user message
            messages.append({"role": "user", "content": user_content})
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract the response content
            response_text = response.choices[0].message.content
            
            # Save response for debugging
            response_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            with open(f"prompts/openai_response_{response_timestamp}.txt", "w", encoding="utf-8") as f:
                f.write(response_text)
            
            return response_text
            
        except Exception as e:
            error_msg = f"Error getting response from OpenAI: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def clean_response(self, character_response):
        """Clean the response and extract chain of thought"""
        # Extract the chain of thought (content between <think> tags)
        chain_of_thought_match = re.search(r'<think>(.*?)</think>', character_response, re.DOTALL)
        chain_of_thought = chain_of_thought_match.group(1).strip() if chain_of_thought_match else ""

        # Remove the <think> content from the character response
        clean_response = re.sub(r'<think>.*?</think>', '', character_response, flags=re.DOTALL).strip()
        
        # Extract character response if enclosed in tags
        char_response_match = re.search(r'<character_response>(.*?)</character_response>', clean_response, re.DOTALL)
        if char_response_match:
            clean_response = char_response_match.group(1).strip()
        
        return clean_response, chain_of_thought
    
    def get_ollama_models(self):
        """
        Get the list of available models from Ollama.
        
        Returns:
            list: A list of available model names.
        """
        try:
            response = requests.get(f"http://{self.ollama_host}:{self.ollama_port}/api/tags")
            response.raise_for_status()  # Raises HTTPError for bad responses
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama: {str(e)}")
            return ["deepseek-coder", "llama2"]  # Return some defaults
    
    def get_openai_models(self):
        """
        Get the list of available models from OpenAI.
        
        Returns:
            list: A list of available model names.
        """
        if not self.openai_client:
            return ["gpt-3.5-turbo", "gpt-4"]  # Default models
        
        try:
            models = self.openai_client.models.list()
            # Filter for relevant models
            gpt_models = [
                model.id for model in models.data 
                if model.id.startswith("gpt-") and not model.id.endswith("-vision-preview")
            ]
            return sorted(gpt_models)
        except Exception as e:
            logger.error(f"Error fetching OpenAI models: {str(e)}")
            return ["gpt-3.5-turbo", "gpt-4"]  # Fallback to default models
    
    def get_available_models(self):
        """Get all available models for current provider"""
        return self.available_models[self.ai_provider]