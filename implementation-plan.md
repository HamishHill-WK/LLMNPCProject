# Implementation Plan: LLM-Enhanced NPCs with Locally-Run Models

## 1. System Architecture

### Core Components

```
┌─────────────────┐     ┌────────────────────┐     ┌────────────────┐
│                 │     │                    │     │                │
│  Game Engine    │◄────┤  NPC Manager       │◄────┤  Local LLM    │
│  (Text-Based)   │     │  (Personality &    │     │  Service       │
│                 │     │   Memory System)   │     │                │
└─────────────────┘     └────────────────────┘     └────────────────┘
```

### Component Breakdown

1. **Game Engine**
   - Text display and input handling
   - Game state management
   - Scene transitions
   - Basic game mechanics

2. **NPC Manager**
   - Character profile database
   - Dialogue history tracking
   - Memory management system
   - Personality enforcement

3. **LLM Service**
   - API connection handling
   - Prompt construction
   - Response parsing
   - Token usage optimization

## 2. Technical Implementation Details

### A. Game Engine (Python with Flask)

```python
# app.py - Main game application
from flask import Flask, render_template, request, session
from npc_manager import NPCManager

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For session management

npc_manager = NPCManager()

@app.route('/', methods=['GET', 'POST'])
def game():
    if 'game_state' not in session:
        initialize_game(session)
    
    if request.method == 'POST':
        player_input = request.form['player_input']
        response = process_player_input(player_input, session)
        return render_template('game.html', response=response, game_state=session['game_state'])
    
    return render_template('game.html', response="Welcome to the game.", game_state=session['game_state'])

def initialize_game(session):
    session['game_state'] = {
        'current_location': 'tavern',
        'current_npc': 'tavernkeeper',
        'inventory': [],
        'quest_status': {},
        'npc_memories': {}
    }

def process_player_input(player_input, session):
    # Basic command parsing (simplified)
    if player_input.startswith('talk '):
        npc_id = player_input[5:]
        session['game_state']['current_npc'] = npc_id
        return f"You approach the {npc_id}."
    elif player_input.startswith('say '):
        message = player_input[4:]
        response = npc_manager.get_npc_response(
            session['game_state']['current_npc'], 
            message,
            session['game_state']
        )
        return response
    
    # Handle other game commands
    return "I don't understand that command."

if __name__ == '__main__':
    app.run(debug=True)
```

### B. NPC Manager System

```python
# npc_manager.py - NPC dialogue and memory management
import json
import os
import time
from llm_service import LocalLLMService

class NPCManager:
    def __init__(self):
        self.characters = self.load_character_profiles()
        # Initialize with a small, low-resource model for development
        self.llm_service = LocalLLMService(
            model_name="models/Llama-2-7B-Chat-GGUF/model-q4_0.gguf", 
            device="cuda" if os.environ.get("USE_GPU", "0") == "1" else "cpu"
        )
        self.memory_keeper = MemorySystem()
        self.cache = ResponseCache()
    
    def load_character_profiles(self):
        characters = {}
        character_files = os.listdir('characters/')
        for file in character_files:
            if file.endswith('.json'):
                with open(f'characters/{file}', 'r') as f:
                    character_data = json.load(f)
                    characters[character_data['character_id']] = character_data
        return characters
    
    def get_npc_response(self, npc_id, player_message, game_state):
        if npc_id not in self.characters:
            return "Error: Character not found."
        
        # Get character profile
        character = self.characters[npc_id]
        
        # Check cache first - for common greetings and responses
        cache_key = f"{npc_id}:{player_message.lower().strip()}"
        cached_response = self.cache.get(cache_key)
        if cached_response:
            print("Using cached response")
            return cached_response
        
        # Retrieve relevant memories
        memories = self.memory_keeper.get_memories(npc_id, game_state)
        
        # Construct prompt with character info and memories
        prompt = self.construct_prompt(character, player_message, memories, game_state)
        
        # Add a progress indicator in the UI
        print(f"Generating response from {character['name']}...")
        
        # Get response from LLM - use lower temperature for more consistent responses
        response = self.llm_service.generate_response(
            prompt, 
            max_new_tokens=100,  # Keep responses shorter for local models
            temperature=0.5      # Lower temperature for more predictable outputs
        )
        
        # Basic response filtering
        response = self.filter_response(response, character)
        
        # Store this interaction in memory
        self.memory_keeper.store_interaction(npc_id, player_message, response, game_state)
        
        # Cache common responses
        if len(player_message) < 20 and player_message.lower() in [
            "hello", "hi", "greetings", "hey", "how are you", 
            "goodbye", "bye", "farewell", "see you", "thanks", "thank you"
        ]:
            self.cache.set(cache_key, response)
        
        return response
    
    def construct_prompt(self, character, player_message, memories, game_state):
        # Optimize prompt for memory-constrained local models
        # Focus on essential character elements and keep context minimal
        
        prompt = f"""You are roleplaying as {character['name']}, a character in a text adventure game.

NAME: {character['name']}

CORE TRAITS: {', '.join(character['core_traits'])}

BACKGROUND: {character['background'][:200]}... (condensed for brevity)

SPEECH PATTERN: {character['speech_pattern']}

{memories[:500] if len(memories) > 500 else memories}

CURRENT SITUATION:
- Location: {game_state['current_location']}
- Player has: {', '.join(game_state['inventory']) if game_state['inventory'] else 'nothing notable'}

The player says to you: "{player_message}"

Respond briefly (1-3 sentences) in character as {character['name']}:"""
        
        return prompt
    
    def filter_response(self, response, character):
        """Apply basic filtering to ensure response quality"""
        # Remove any preamble like "As [character name]" that models often add
        prefixes_to_remove = [
            f"As {character['name']},", 
            f"{character['name']}:",
            "I respond,",
            "I say,",
            "*Roleplaying as",
            "*In character*"
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # If response is too long, truncate to complete sentences
        if len(response) > 300:
            sentences = response.split('.')
            truncated = []
            total_length = 0
            
            for sentence in sentences:
                if total_length + len(sentence) < 300:
                    truncated.append(sentence)
                    total_length += len(sentence) + 1  # +1 for the period
                else:
                    break
            
            response = '.'.join(truncated) + '.'
        
        return response

class ResponseCache:
    """Simple cache for common NPC responses to improve performance"""
    def __init__(self, max_size=100, ttl=3600):  # 1 hour TTL
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key):
        """Get a cached response if available and not expired"""
        if key in self.cache:
            # Check if entry has expired
            if time.time() - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            return self.cache[key]
        return None
    
    def set(self, key, value):
        """Add a response to the cache"""
        # If cache is full, remove oldest entry
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
```

### C. Memory System

```python
# Part of npc_manager.py
class MemorySystem:
    def __init__(self):
        self.memory_file = 'data/memories.json'
        self.memories = self.load_memories()
        self.max_memories = 10  # Keep last N interactions per NPC
    
    def load_memories(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_memories(self):
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, 'w') as f:
            json.dump(self.memories, f)
    
    def get_memories(self, npc_id, game_state):
        if npc_id not in self.memories:
            return "No previous interactions."
        
        memory_strings = []
        for memory in self.memories[npc_id][-5:]:  # Last 5 memories
            memory_strings.append(f"Player: {memory['player_message']}\n{npc_id.capitalize()}: {memory['response']}")
        
        return "\n\n".join(memory_strings)
    
    def store_interaction(self, npc_id, player_message, response, game_state):
        if npc_id not in self.memories:
            self.memories[npc_id] = []
        
        self.memories[npc_id].append({
            'player_message': player_message,
            'response': response,
            'game_state': {
                'location': game_state['current_location'],
                'inventory': game_state['inventory'].copy(),
                'timestamp': self.get_timestamp()
            }
        })
        
        # Trim to keep only most recent N memories
        if len(self.memories[npc_id]) > self.max_memories:
            self.memories[npc_id] = self.memories[npc_id][-self.max_memories:]
        
        self.save_memories()
    
    def get_timestamp(self):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

### D. Local LLM Service

```python
# llm_service.py - Connection to locally-hosted LLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer
from threading import Thread

class LocalLLMService:
    def __init__(self, model_name="TheBloke/Llama-2-7B-Chat-GGUF", device="cpu"):
        """
        Initialize the local LLM service.
        
        Args:
            model_name (str): The model identifier from Hugging Face or local path
            device (str): "cpu" or "cuda" (if GPU is available)
        """
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model_name = model_name
        
        # Load model based on available hardware
        self.setup_model()
    
    def setup_model(self):
        """Load the model based on available hardware resources"""
        print(f"Initializing model {self.model_name} on {self.device}...")
        
        try:
            # For GGUF quantized models with llama.cpp (lowest resource usage)
            from ctransformers import AutoModelForCausalLM as CTAutoModelForCausalLM
            
            self.model = CTAutoModelForCausalLM.from_pretrained(
                self.model_name,
                model_type="llama",
                gpu_layers=0 if self.device == "cpu" else 50  # Adjust based on GPU VRAM
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.use_ctransformers = True
            print("Using llama.cpp backend (CTTransformers)")
            
        except (ImportError, ValueError, Exception) as e:
            print(f"Could not load with CTTransformers: {e}")
            print("Falling back to Transformers library")
            
            # Try standard Hugging Face Transformers approach
            try:
                # Use 4-bit or 8-bit quantization to reduce memory usage
                from transformers import BitsAndBytesConfig
                
                if torch.cuda.is_available() and self.device == "cuda":
                    # 4-bit quantization for GPU
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                else:
                    quantization_config = None
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
                self.use_ctransformers = False
                
            except Exception as e:
                print(f"Error loading model with Transformers: {e}")
                raise RuntimeError("Could not initialize any model backend")
        
        print(f"Model loaded successfully with {'llama.cpp' if hasattr(self, 'use_ctransformers') and self.use_ctransformers else 'transformers'}")
    
    def generate_response(self, prompt, max_new_tokens=150, temperature=0.7):
        """Generate a response from the local model"""
        try:
            if hasattr(self, 'use_ctransformers') and self.use_ctransformers:
                # CTTransformers generation
                input_text = self._format_prompt_for_model(prompt)
                
                # Generate text with llama.cpp backend
                response = self.model(
                    input_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                
                # Extract just the assistant's response
                return self._extract_response(input_text, response)
            
            else:
                # Standard Transformers generation
                inputs = self.tokenizer(self._format_prompt_for_model(prompt), return_tensors="pt").to(self.device)
                
                # Create a streamer for non-blocking generation
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                
                # Start generation in a separate thread
                generation_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "streamer": streamer,
                }
                
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()
                
                # Collect the generated text
                generated_text = ""
                for text in streamer:
                    generated_text += text
                
                return generated_text.strip()
                
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I cannot respond right now."
    
    def _format_prompt_for_model(self, prompt):
        """Format the prompt based on the model type"""
        # Format for Llama-2-Chat models
        if "llama-2" in self.model_name.lower() and "chat" in self.model_name.lower():
            return f"<s>[INST] {prompt} [/INST]"
        
        # Format for general instruction models
        return f"### Instruction:\n{prompt}\n\n### Response:"
    
    def _extract_response(self, prompt, full_response):
        """Extract just the model's response part from the full response"""
        # Remove the original prompt to get just the generated text
        if full_response.startswith(prompt):
            response = full_response[len(prompt):].strip()
        else:
            response = full_response
            
        # Remove any additional prompt markers that might have been generated
        end_markers = ["[INST]", "### Instruction:", "<|user|>", "<s>"]
        for marker in end_markers:
            if marker in response:
                response = response.split(marker)[0].strip()
                
        return response

    def get_model_info(self):
        """Return information about the currently loaded model"""
        device_str = f"GPU ({torch.cuda.get_device_name(0)})" if self.device == "cuda" else "CPU"
        backend = "llama.cpp (CTTransformers)" if hasattr(self, 'use_ctransformers') and self.use_ctransformers else "Hugging Face Transformers"
        
        return {
            "model_name": self.model_name,
            "device": device_str,
            "backend": backend,
        }
```

## 3. Character Profile Structure

Create JSON files for each character with this structure:

```json
{
  "character_id": "tavernkeeper",
  "name": "Greta",
  "core_traits": [
    "gruff but fair",
    "efficient",
    "protective of establishment",
    "values honesty"
  ],
  "background": "Former soldier who fought in the Northern Wars. Runs the tavern for 15 years since retiring from the army. Lost family during the war and considers the tavern patrons her new family.",
  "speech_pattern": "Short sentences. Northern dialect. Uses 'aye' and 'nay'. Rarely uses pleasantries or small talk. Often uses metaphors related to battle or survival.",
  "knowledge_boundaries": [
    "Knows local town gossip and politics",
    "Familiar with basic regional history and trade routes",
    "Understands military tactics and weapons",
    "No knowledge of magic or distant kingdoms"
  ],
  "goals": [
    "Keep tavern profitable and respected",
    "Protect regular customers from trouble",
    "Maintain order in her establishment",
    "Avoid entanglements with nobility or officials"
  ],
  "relationships": {
    "village_elder": "Respectful but cautious",
    "blacksmith": "Good friends and drinking buddies",
    "mysterious_stranger": "Deeply suspicious and watchful"
  },
  "system_prompt": "You are role-playing as Greta, the tavernkeeper in a medieval fantasy setting. You must maintain your established personality traits and background in all interactions. Never break character. Your responses should reflect your gruff nature and military background. Keep your responses concise and practical, focused on the immediate situation and your tavern business. Never use modern language or references."
}
```

## 4. Personality Consistency Enforcement

Enhance the NPC Manager with a consistency checker:

```python
# Add to npc_manager.py
def validate_response_consistency(self, character, response, game_state):
    # Check for personality trait consistency
    trait_consistency = self.check_trait_alignment(character['core_traits'], response)
    
    # Check for speech pattern consistency
    speech_consistency = self.check_speech_pattern(character['speech_pattern'], response)
    
    # If response seems inconsistent, regenerate or modify
    if trait_consistency < 0.7 or speech_consistency < 0.7:
        # Option 1: Add more character context and regenerate
        enhanced_prompt = self.construct_enhanced_prompt(character, response, game_state)
        return self.llm_service.generate_response(enhanced_prompt)
    
    return response

def check_trait_alignment(self, traits, response):
    # In a real implementation, this could use:
    # 1. Another LLM call to analyze alignment
    # 2. Keyword/sentiment matching
    # 3. Embedding similarity to exemplar responses
    
    # Simplified version for prototype
    trait_keywords = {
        "gruff": ["direct", "blunt", "terse", "short", "grumble"],
        "fair": ["honest", "equal", "fair", "just", "reasonable"],
        "efficient": ["quick", "efficient", "prompt", "direct", "straightforward"],
        "protective": ["watch", "careful", "protect", "safe", "guard"]
    }
    
    score = 0
    for trait, keywords in trait_keywords.items():
        if any(keyword in response.lower() for keyword in keywords):
            score += 1
    
    return score / len(trait_keywords)

def construct_enhanced_prompt(self, character, original_response, game_state):
    # Create a more detailed prompt with explicit consistency guidance
    prompt = f"""
    Your previous response as {character['name']} was not fully consistent with the character's personality.
    
    Character traits to emphasize:
    {', '.join(character['core_traits'])}
    
    Speech pattern to maintain:
    {character['speech_pattern']}
    
    Your previous response: "{original_response}"
    
    Please revise to better match {character['name']}'s personality and speech pattern.
    """
    return prompt
```

## 5. Memory Enhancement System

Extend the memory system with summarization and importance ranking:

```python
# Add to memory_system.py
def summarize_memories(self, npc_id):
    """Periodically summarize older memories to prevent context window overflow"""
    if npc_id not in self.memories or len(self.memories[npc_id]) < 20:
        return
    
    # Group memories to summarize (e.g., memories 10-20)
    memories_to_summarize = self.memories[npc_id][10:20]
    
    # Create a summary prompt
    summary_prompt = f"""
    Summarize the following conversation between the player and {npc_id} into 1-2 key points:
    
    {"".join([f"Player: {m['player_message']}\n{npc_id.capitalize()}: {m['response']}\n" for m in memories_to_summarize])}
    """
    
    # Generate summary using LLM
    summary = self.llm_service.generate_response(summary_prompt)
    
    # Replace the detailed memories with the summary
    summary_memory = {
        'player_message': '[SUMMARY]',
        'response': summary,
        'game_state': memories_to_summarize[-1]['game_state'],
        'is_summary': True
    }
    
    # Replace multiple memories with single summary
    self.memories[npc_id] = self.memories[npc_id][:10] + [summary_memory] + self.memories[npc_id][20:]
    self.save_memories()

def rank_memory_importance(self, npc_id, memory):
    """Assign importance score to new memories for retention decisions"""
    # Factors that might make a memory important:
    # - Contains player promises or commitments
    # - Reveals player background information
    # - Contains emotional exchanges
    # - Mentions key game items or characters
    # - Involves quest information
    
    importance_prompt = f"""
    On a scale of 1-10, how important is this exchange to remember for future interactions?
    Consider if it contains promises, personal revelations, emotional moments, or plot-relevant information.
    
    Player: {memory['player_message']}
    {npc_id.capitalize()}: {memory['response']}
    
    Respond with just a number 1-10:
    """
    
    try:
        importance = int(self.llm_service.generate_response(importance_prompt).strip())
        return min(max(importance, 1), 10)  # Ensure between 1-10
    except:
        return 5  # Default medium importance
```

## 6. Web Interface (Simplified HTML/CSS)

```html
<!-- templates/game.html -->
<!DOCTYPE html>
<html>
<head>
    <title>LLM-Enhanced Text Adventure</title>
    <style>
        body {
            font-family: monospace;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #222;
            color: #ccc;
        }
        #game-output {
            background-color: #333;
            border: 1px solid #555;
            padding: 15px;
            height: 400px;
            overflow-y: scroll;
            margin-bottom: 10px;
            white-space: pre-wrap;
        }
        #input-form {
            display: flex;
        }
        #player-input {
            flex-grow: 1;
            background-color: #333;
            border: 1px solid #555;
            padding: 8px;
            color: #ccc;
        }
        button {
            background-color: #555;
            color: #ccc;
            border: none;
            padding: 8px 15px;
            margin-left: 5px;
        }
        .npc-response {
            color: #88cc88;
        }
        .player-input {
            color: #8888ff;
        }
        .system-message {
            color: #cccc88;
            font-style: italic;
        }
    </style>
</head>
<body>
    <h1>Text Adventure</h1>
    <div id="game-output">
        <div class="system-message">Welcome to the Tavern of the Forgotten Hero...</div>
        <div class="npc-response">{{ response }}</div>
    </div>
    <form id="input-form" method="post">
        <input type="text" id="player-input" name="player_input" placeholder="What will you do?" autofocus>
        <button type="submit">Submit</button>
    </form>
    <div>
        <p>Type 'help' for commands</p>
    </div>
    
    <script>
        // Simple JavaScript to add player input to the display
        document.getElementById('input-form').addEventListener('submit', function(e) {
            const input = document.getElementById('player-input').value;
            const outputDiv = document.getElementById('game-output');
            
            // Add player input to the display
            const playerDiv = document.createElement('div');
            playerDiv.className = 'player-input';
            playerDiv.textContent = '> ' + input;
            outputDiv.appendChild(playerDiv);
            
            // Auto-scroll to bottom
            outputDiv.scrollTop = outputDiv.scrollHeight;
        });
    </script>
</body>
</html>
```

## 7. Installation and Setup Instructions

### Prerequisites
- Python 3.8+ installed
- Flask and required packages
- LLM API access (OpenAI, Anthropic, etc.)

### Step-by-Step Setup

1. **Create Project Structure**

```
text_adventure/
├── app.py
├── npc_manager.py
├── llm_service.py
├── requirements.txt
├── characters/
│   ├── tavernkeeper.json
│   ├── blacksmith.json
│   └── mysterious_stranger.json
├── data/
│   └── memories.json
└── templates/
    └── game.html
```

2. **Install Dependencies**

```
pip install -r requirements.txt
```

Contents of requirements.txt:
```
flask==2.0.1
requests==2.26.0
python-dotenv==0.19.0
```

3. **Environment Setup**

Create a .env file with your API keys:
```
LLM_API_KEY=your_openai_or_anthropic_key_here
LLM_API_URL=https://api.openai.com/v1/chat/completions
```

4. **Run the Application**

```
python app.py
```

Access the game at http://localhost:5000

## 8. Testing Strategy

### Unit Tests

Create tests for core functionality:

```python
# tests/test_npc_manager.py
import unittest
from npc_manager import NPCManager, MemorySystem

class TestNPCManager(unittest.TestCase):
    def setUp(self):
        self.npc_manager = NPCManager()
        
    def test_character_loading(self):
        # Test characters are properly loaded
        self.assertIn('tavernkeeper', self.npc_manager.characters)
        
    def test_prompt_construction(self):
        # Test prompt includes character traits
        character = self.npc_manager.characters['tavernkeeper']
        game_state = {'current_location': 'tavern', 'inventory': []}
        prompt = self.npc_manager.construct_prompt(
            character, 
            "Hello there", 
            "No previous interactions.", 
            game_state
        )
        
        for trait in character['core_traits']:
            self.assertIn(trait, prompt)
        
    # Add more tests for other functions
```

### Integration Tests

Test the full conversation flow with mock LLM responses:

```python
# tests/test_integration.py
import unittest
from unittest.mock import patch
from app import app

class TestGameIntegration(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        
    @patch('llm_service.LLMService.generate_response')
    def test_conversation_flow(self, mock_generate):
        # Mock LLM response
        mock_generate.return_value = "Aye, what can I get for ye?"
        
        # Start game session
        response = self.client.get('/')
        self.assertIn('Welcome', response.data.decode())
        
        # Test NPC conversation
        response = self.client.post('/', data={
            'player_input': 'say Hello, tavernkeeper'
        }, follow_redirects=True)
        
        self.assertIn('Aye, what can I get for ye?', response.data.decode())
```

## 9. Local LLM Optimization Strategies

### Hardware Optimization

1. **Model Quantization**
   - Use 4-bit or 8-bit quantized models (GGUF format)
   - Balance quality vs. performance based on available hardware
   - Test different quantization levels for your specific use case

2. **CPU Optimization**
   - Set appropriate thread count based on your CPU
   - Use llama.cpp backend for optimal CPU performance
   - Consider batching multiple character responses during idle times

3. **GPU Acceleration**
   - Use CUDA for NVIDIA GPUs when available
   - Adjust layers offloaded to GPU based on VRAM available
   - Monitor GPU temperature during extended sessions

### Response Quality Optimization

1. **Prompt Engineering for Local Models**
   - Keep prompts significantly shorter than for API models
   - Focus on essential character information
   - Include explicit formatting instructions
   - Use few-shot examples for consistent outputs

2. **Response Processing**
   - Implement more aggressive response filtering
   - Consider rule-based post-processing for common issues
   - Have fallback canned responses for low-quality outputs

3. **Performance Strategies**
   - Implement aggressive caching for common interactions
   - Pre-generate responses for predictable scenarios
   - Use smaller models for background NPCs, larger for key characters

### Memory Management

1. **Context Window Management**
   - Limit memory inclusion to 2-3 recent interactions
   - Use aggressive summarization techniques
   - Prioritize context based on conversation relevance

2. **Progressive Loading**
   - Start with smaller models during development
   - Use placeholder responses during initial testing
   - Implement model switching based on scene importance

### User Experience Considerations

1. **Response Time Management**
   - Add "typing" indicators during generation
   - Break longer responses into chunks delivered progressively 
   - Consider pre-loading common responses at scene transitions

2. **Failure Handling**
   - Create robust fallback mechanisms
   - Add pre-written responses for critical interactions
   - Implement graceful degradation when resources are constrained

## 10. Extensibility Options

### A. Multiple NPC Conversations

Extend the system to handle multiple NPCs in the same scene:

```python
def process_group_conversation(self, player_message, npc_ids, game_state):
    # Determine which NPC should respond
    target_npc = self.determine_responder(player_message, npc_ids, game_state)
    
    # Get response from that NPC
    response = self.get_npc_response(target_npc, player_message, game_state)
    
    # Update other NPCs' knowledge of this interaction
    for npc_id in npc_ids:
        if npc_id != target_npc:
            self.memory_keeper.observe_interaction(
                npc_id, target_npc, player_message, response, game_state
            )
    
    return target_npc, response

def determine_responder(self, player_message, npc_ids, game_state):
    # Simple version: Check if message directly addresses an NPC
    for npc_id in npc_ids:
        character = self.characters[npc_id]
        if character['name'].lower() in player_message.lower():
            return npc_id
    
    # Otherwise, determine based on relevance to each NPC's interests/knowledge
    # This could be done with a simple LLM call
    relevance_prompt = f"""
    The player says: "{player_message}"
    
    Which of these NPCs would most likely respond?
    {', '.join([self.characters[npc_id]['name'] for npc_id in npc_ids])}
    
    Choose just one name:
    """
    
    response = self.llm_service.generate_response(relevance_prompt).strip()
    
    # Map response back to npc_id
    for npc_id in npc_ids:
        if self.characters[npc_id]['name'].lower() in response.lower():
            return npc_id
    
    # Default to first NPC if no match
    return npc_ids[0]
```

### B. Emotional State Tracking

Add emotional state tracking to enhance personality consistency:

```python
# Add to character profiles
"emotional_states": {
    "neutral": "Your default state. You are businesslike and focused.",
    "angry": "You speak in shorter sentences. Your military background shows in your threats.",
    "friendly": "You are still direct, but use warmer language and may offer small discounts.",
    "suspicious": "You ask more questions and reveal less information. You watch carefully."
}

# Add to NPC manager
def update_emotional_state(self, npc_id, player_message, current_state):
    emotion_prompt = f"""
    The player says to {self.characters[npc_id]['name']}: "{player_message}"
    
    Given this interaction, which emotional state would {self.characters[npc_id]['name']} most likely shift to?
    Current state: {current_state}
    
    Possible states: {', '.join(self.characters[npc_id]['emotional_states'].keys())}
    
    Respond with just one word - the emotional state:
    """
    
    new_state = self.llm_service.generate_response(emotion_prompt).strip().lower()
    
    # Verify it's a valid state
    if new_state in self.characters[npc_id]['emotional_states']:
        return new_state
    else:
        return current_state  # No change if invalid
```

### C. Quest System Integration

```python
# Add to game_state
'quests': {
    'missing_sword': {
        'status': 'active',
        'description': 'Find the blacksmith\'s missing sword',
        'npc_knowledge': {
            'tavernkeeper': 'Has heard rumors about thieves',
            'blacksmith': 'Knows the sword was stolen two nights ago',
            'mysterious_stranger': 'Has the sword hidden in their room'
        }
    }
}

# Modify prompt construction to include relevant quest info
def construct_prompt(self, character, player_message, memories, game_state):
    # Base prompt construction as before
    
    # Add relevant quest knowledge
    quest_knowledge = []
    for quest_id, quest in game_state['quests'].items():
        if quest['status'] == 'active' and character['character_id'] in quest['npc_knowledge']:
            quest_knowledge.append(f"About the quest '{quest['description']}': {quest['npc_knowledge'][character['character_id']]}")
    
    if quest_knowledge:
        prompt += f"\n\nYou know the following about active quests:\n{chr(10).join(quest_knowledge)}"
    
    return prompt
```
