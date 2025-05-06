from flask import Flask, render_template, request, jsonify, Response
import webbrowser
import os
import prompt_engine as pe
import json
import memory_manager
import ollama_manager as om
import re
import knowledge_engine as ke
import executive as kep
import logging
import time
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Set up logger
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))
handler = logging.StreamHandler()
logger.addHandler(handler)

# Flask application setup
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your_secret_key")

# Cache configuration
MAX_CACHE_SIZE = int(os.environ.get("MAX_CACHE_SIZE", "100"))
CACHE_EXPIRY_TIME = int(os.environ.get("CACHE_EXPIRY_TIME", "60"))

# Models from compose.yaml to track
TRACKED_MODELS = ['deepseek-r1:8b', 'deepseek-r1:1.5b', 'llama2:7b']
model_status = {model: {'downloaded': False, 'in_progress': False, 'error': None} for model in TRACKED_MODELS}

# Game state defaults
DEFAULT_LOCATION = os.environ.get("DEFAULT_LOCATION", "tavern")
DEFAULT_NPC = os.environ.get("DEFAULT_NPC", "tavernkeeper")

# Initialize engines and managers
knowledge_engine = ke.KnowledgeEngine()
executive_planner = kep.KnowledgeExecutivePlanner(knowledge_engine=knowledge_engine)
Mem_manager = memory_manager.MemoryManager(
    max_short_term=int(os.environ.get("MAX_SHORT_TERM_MEMORY", "3")),
    max_long_term=int(os.environ.get("MAX_LONG_TERM_MEMORY", "3")), 
    knowledge_engine=knowledge_engine
)
prompt_engine = pe.Prompt_Engine(memory_manager=Mem_manager, knowledge_engine=knowledge_engine)

# Initialize AI manager
ollama_manager = om.OllamaManager(prompt_engine=prompt_engine)

# AI service configuration
ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

ai_config = {
    'provider': os.environ.get("AI_PROVIDER", "ollama"),
    'model': os.environ.get("DEFAULT_OLLAMA_MODEL", "deepseek-r1:8b"),
    'openai_model': os.environ.get("DEFAULT_OPENAI_MODEL", "gpt-3.5-turbo"),
    'available_models': {
        'ollama': ollama_manager.available_models.get('ollama', []),
        'openai': ollama_manager.available_models.get('openai', [])
    }
}

# Simple game state
game_state = {
    'current_location': DEFAULT_LOCATION,
    'current_npc': DEFAULT_NPC,
    'all_characters': [],
    'inventory': [],
    'ai_config': ai_config
}

# Simulation state
simulation_state = {
    'current_location': DEFAULT_LOCATION,
    'current_speaker' : 'npc_A',
    'current_listener' : 'npc_B',
    'npc_A': DEFAULT_NPC,
    'npc_B': os.environ.get("DEFAULT_SIM_NPC", "blacksmith"),
    'initial_prompt': 'You are in a tavern. The tavernkeeper greets you with a smile.',
    'all_characters': [],
    'inventory': [],
    'ai_config': ai_config
}

# Request deduplication cache
request_cache = {}
MAX_CACHE_SIZE = 100  # Maximum number of request IDs to cache
CACHE_EXPIRY_TIME = 60  # Seconds before a cached request can be removed

# Cleanup old entries in the request cache
def cleanup_request_cache():
    current_time = time.time()
    expired_keys = [k for k, v in request_cache.items() if current_time - v > CACHE_EXPIRY_TIME]
    
    for key in expired_keys:
        del request_cache[key]
    
    # If cache is still too large, remove oldest entries
    if len(request_cache) > MAX_CACHE_SIZE:
        sorted_keys = sorted(request_cache.keys(), key=lambda k: request_cache[k])
        for key in sorted_keys[:len(request_cache) - MAX_CACHE_SIZE]:
            del request_cache[key]

@app.route('/', methods=['GET', 'POST'])
def game():
    response = "Welcome to the Text Adventure Game. Type something to get started."
    
    if request.method == 'POST':
        player_input = request.form.get('player_input', '')
        response = f"You said: {player_input}"
        
    # Add AI provider and models to game state
    game_state['ai_provider'] = ollama_manager.ai_provider
    game_state['available_models'] = ollama_manager.available_models
    game_state['selected_model'] = ollama_manager.selected_model
        
    return render_template('game.html', response=response, game_state=game_state, simulation_state=simulation_state)

@app.route('/api/change_npc', methods=['POST'])
def change_npc():
    try:
        data = request.json
        if not data or 'npc_id' not in data:
            return jsonify({"error": "Missing NPC ID"}), 400
            
        npc_id = data['npc_id']
        
        game_state['current_npc'] = npc_id
        
        return jsonify({
            "game_state": game_state
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/execute_ollama_command', methods=['POST'])
def execute_ollama_command():
    """Execute an Ollama command from the UI"""
    try:
        data = request.json
        if not data or 'command' not in data:
            return jsonify({"success": False, "error": "Missing command"}), 400
            
        command = data['command']
        logger.info(f"Executing Ollama command: {command}")
        
        # Security check - only allow specific commands
        if not command.startswith(('ollama pull', 'ollama run', 'ollama list')):
            return jsonify({
                "success": False,
                "error": "Only 'ollama pull', 'ollama run', and 'ollama list' commands are allowed"
            }), 400
        
        # Execute the command
        import subprocess
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate()
        
        # Check if the command was successful
        if process.returncode == 0:
            # Refresh available models list if it was a pull command
            if command.startswith('ollama pull'):
                ollama_manager.available_models['ollama'] = ollama_manager.get_ollama_models()
                game_state['available_models'] = ollama_manager.available_models
            
            return jsonify({
                "success": True,
                "output": stdout,
                "available_models": ollama_manager.available_models
            })
        else:
            return jsonify({
                "success": False,
                "error": stderr or "Command failed"
            }), 500
            
    except Exception as e:
        logger.error(f"Error executing Ollama command: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/change_sim_npc', methods=['POST'])
def change_simulation_npc():
    try:
        data = request.json
        if not data or 'npc_id' not in data:
            return jsonify({"error": "Missing NPC ID"}), 400
            
        npc_id = data['npc_id']
        
        simulation_state[f"npc_{data['target_id']}"] = npc_id

        return jsonify({
            "game_state": game_state
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/simulate_stream', methods=['GET'])
def simulate_conversation():
    npc_a = request.args.get('npc_a')
    npc_b = request.args.get('npc_b')
    simulation_state['npc_A'] = npc_a
    simulation_state['npc_B'] = npc_b
    initial_prompt = request.args.get('simulation_input')
    turns = int(request.args.get('turns', 5))
    
    def generate():
        # Send SSE headers
        yield "event: start\ndata: Conversation starting\n\n"
        
        # Get initial message
        message = initial_prompt
        
        # Simulate conversation turns
        for i in range(turns):            
            # Set up current speakers
            current_speaker = simulation_state['current_speaker']
            current_listener = simulation_state['current_listener']
            speaker_npc = simulation_state[current_speaker]
            listener_npc = simulation_state[current_listener]
            
            # Get conversation context for current listener
            conversation_context = Mem_manager.get_character_memory(
                simulation_state['all_characters'], 
                listener_npc
            )
            
            # Use the enhanced ollama_manager which now handles both providers
            
            # Set model based on current provider
            model_name = (ollama_manager.selected_model if ollama_manager.ai_provider == 'ollama' 
                         else "gpt-3.5-turbo" if ollama_manager.ai_provider == 'openai' else "deepseek-r1:8b")
            
            # Create data for first response
            data = {
                "model": model_name,
                "prompt": message,
                "stream": False,
                "max_tokens": 150
            }
            
            # Use executive planner to analyze if knowledge is needed
            knowledge_analysis = executive_planner.analyze_for_knowledge(
                player_input=message,
                character_id=listener_npc,
                game_state=simulation_state,
                conversation_context=conversation_context,
            )
            
            # Assess and gather knowledge
            knowledge_engine.assess_knowledge(
                player_input=message,
                character_id=listener_npc,
                conversation_context=conversation_context,
                data_dict=data,
                ollama_service=ollama_manager
            )
            
            # Add relevant knowledge if 
            if "knowledge_required" in knowledge_analysis:
                if knowledge_analysis.get('knowledge_required', False) and 'memory_search_keywords' in knowledge_analysis:
                    data['relevant_knowledge'] = knowledge_engine.search_knowledge_base(
                        knowledge_analysis['memory_search_keywords']
                    )
                
            # Generate first NPC response
            npc1_response = ollama_manager.get_response(data, simulation_state, Mem_manager)
            npc1_response_clean, chain_of_thought1 = ollama_manager.clean_response(npc1_response)
            
            # Store interaction in memory with proper chain of thought
            Mem_manager.add_interaction(
                listener_npc, 
                speaker_npc, 
                message, 
                npc1_response_clean, 
                chain_of_thought1, 
                simulation_state['current_location']
            )
            
            # Switch speaker and listener
            simulation_state['current_listener'] = simulation_state['current_speaker']
            simulation_state['current_speaker'] = 'npc_A' if simulation_state['current_speaker'] == 'npc_B' else 'npc_B'
            
            # Send response to client
            yield f"data: {json.dumps({'speaker': listener_npc, 'message': npc1_response_clean, 'turn': i * 2})}\n\n"
            
            # Update current speakers for second response
            current_speaker = simulation_state['current_speaker']
            current_listener = simulation_state['current_listener']
            speaker_npc = simulation_state[current_speaker]
            listener_npc = simulation_state[current_listener]
            
            # Get conversation context for second NPC
            conversation_context2 = Mem_manager.get_character_memory(
                simulation_state['all_characters'], 
                listener_npc
            )
            
            # Create data for second response
            data2 = {
                "model": model_name,  # Use the same model as the first response
                "prompt": npc1_response_clean,
                "stream": False,
                "max_tokens": 150
            }
            
            # Analyze knowledge needs for second response
            knowledge_analysis2 = executive_planner.analyze_for_knowledge(
                player_input=npc1_response_clean,
                character_id=listener_npc,
                game_state=simulation_state,
                conversation_context=conversation_context2,
            )
            
            # Assess and gather knowledge for second response
            knowledge_engine.assess_knowledge(
                player_input=npc1_response_clean,
                character_id=listener_npc,
                conversation_context=conversation_context2,
                data_dict=data2,
                ollama_service=ollama_manager  # Use the enhanced ollama_manager
            )
            
            # Add relevant knowledge if needed
            if "knowledge_required" in knowledge_analysis2:
                if knowledge_analysis2.get('knowledge_required', False) and 'memory_search_keywords' in knowledge_analysis2:
                    data2['relevant_knowledge'] = knowledge_engine.search_knowledge_base(
                        knowledge_analysis2['memory_search_keywords']
                    )
                
            # Generate second NPC response
            npc2_response = ollama_manager.get_response(data2, simulation_state, Mem_manager)
            npc2_response_clean, chain_of_thought2 = ollama_manager.clean_response(npc2_response)
            
            # Store second interaction in memory
            Mem_manager.add_interaction(
                listener_npc,
                speaker_npc,
                npc1_response_clean,
                npc2_response_clean,
                chain_of_thought2,
                simulation_state['current_location']
            )
            
            # Switch speaker and listener again
            simulation_state['current_listener'] = simulation_state['current_speaker']
            simulation_state['current_speaker'] = 'npc_A' if simulation_state['current_speaker'] == 'npc_B' else 'npc_B'
            
            # Send second response to client
            yield f"data: {json.dumps({'speaker': listener_npc, 'message': npc2_response_clean, 'turn': i * 2 + 1})}\n\n"
            
            # Update message for next turn
            message = npc2_response_clean
            
        yield "event: end\ndata: Conversation complete\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/interact', methods=['POST'])
def api_interact():
    """API endpoint for AJAX interactions"""
    try:
        data = request.json
        if not data or 'player_input' not in data:
            return jsonify({"error": "Missing player input"}), 400
            
        player_input = data['player_input']
      
        prompt = player_input.strip()
        data = {
            "model": ollama_manager.selected_model,
            "prompt": prompt,
            "stream": False,
            "max_tokens": 150
        }
        
        # Check for request_id for deduplication
        request_id = data.get('request_id')
        if request_id:
            # Check if this request has been seen recently
            if request_id in request_cache:
                logger.info(f"Duplicate request detected with ID: {request_id}")
                return jsonify({"response": "Your message is being processed, please wait...", 
                               "note": "Duplicate request detected"})
            
            # Add to cache with timestamp
            request_cache[request_id] = time.time()
            
            # Clean up old cache entries periodically
            cleanup_request_cache()
        
        # Get conversation context for the character
        conversation_context = Mem_manager.get_character_memory(
            game_state['all_characters'], 
            game_state['current_npc']
        )
        
        # Use the executive planner to analyze if knowledge is needed
        knowledge_analysis = executive_planner.analyze_for_knowledge(
            player_input=player_input,
            character_id=game_state['current_npc'],
            game_state=game_state,
            conversation_context=conversation_context,
        )

        knowledge_engine.assess_knowledge(
            player_input=player_input,
            character_id=game_state['current_npc'],
            conversation_context=conversation_context,
            data_dict=data,
            ollama_service=ollama_manager  
        )
        
        response = ""
        
        # if len(knowledge_analysis['message_types']) == 1 and ('greeting' in knowledge_analysis['message_types'] or 'farewell' in knowledge_analysis['message_types']):
        #     response = ollama_manager.get_response(data, game_state, Mem_manager)
        #     response, chain_of_thought = ollama_manager.clean_response(response)
        #     Mem_manager.add_interaction(game_state['current_npc'], "Player", player_input, response, chain_of_thought, game_state['current_location'])

        #print(knowledge_analysis)

        if 'knowledge_query' in knowledge_analysis:
            #print("Knowledge query detected")
            if knowledge_analysis['knowledge_required'] and 'memory_search_keywords' in knowledge_analysis:
                data['relevant_knowledge'] = knowledge_engine.search_knowledge_base(knowledge_analysis['memory_search_keywords'])
            
            response = ollama_manager.get_response(data, game_state, Mem_manager)
            response, chain_of_thought = ollama_manager.clean_response(response)
            Mem_manager.add_interaction(game_state['current_npc'], "Player", player_input, response, chain_of_thought, game_state['current_location'])
                
        elif 'requires_memory' in knowledge_analysis:
            #print("Memory required")
            if knowledge_analysis['requires_memory'] and 'memory_search_keywords' in knowledge_analysis:
                data['relevant_knowledge'] = knowledge_engine.search_knowledge_base(knowledge_analysis['memory_search_keywords'])
            
            response = ollama_manager.get_response(data, game_state, Mem_manager)
            #print("Response:", response) 
            response, chain_of_thought = ollama_manager.clean_response(response)
            Mem_manager.add_interaction(game_state['current_npc'], "Player", player_input, response, chain_of_thought, game_state['current_location'])
                
        else:
            #print("No knowledge query or memory required, using default response")
            response = ollama_manager.get_response(data, game_state, Mem_manager)
            response, chain_of_thought = ollama_manager.clean_response(response)
            Mem_manager.add_interaction(game_state['current_npc'], "Player", player_input, response, chain_of_thought, game_state['current_location'])
        
        # Always return a response, with any additional data as needed
        return jsonify({
            "response": response,
            "game_state": game_state,
            "knowledge_used": 'knowledge_query' in knowledge_analysis and knowledge_analysis.get('knowledge_required', False)
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/change_ai_config', methods=['POST'])
def change_ai_config():
    """API endpoint to change the AI provider, model, and API key"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Missing data"}), 400
            
        success = False
        
        # Extract parameters
        provider = data.get('provider')
        model = data.get('model')
        api_key = data.get('api_key')
        
        # Update AI provider and model in ollama_manager
        success = ollama_manager.set_ai_provider(provider, model, api_key)
        
        # Update executive_planner and Mem_manager to use the updated ollama_manager
        # This ensures they use the correct AI provider
        executive_planner.set_ollama(ollama_manager)
        Mem_manager.set_ollama(ollama_manager)
        
        # Return the updated configuration
        return jsonify({
            "success": success,
            "ai_provider": ollama_manager.ai_provider,
            "selected_model": ollama_manager.selected_model,
            "available_models": ollama_manager.available_models
        })
    except Exception as e:
        print(f"Error changing AI config: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear_memories', methods=['POST'])
def clear_memories():
    """API endpoint to clear all character memories"""
    try:
        # Load the default memories from memories_default.json
        default_memories_path = 'data/memories_default.json'
        if os.path.exists(default_memories_path):
            with open(default_memories_path, 'r') as f:
                default_memories = json.load(f)
            
            # Replace the current memories with defaults
            Mem_manager.memories = default_memories
        else:
            # If default file doesn't exist, just clear the memories
            Mem_manager.memories = {}
        
        # Save the reset memories state to file
        Mem_manager.save_memories()
        
        return jsonify({
            "success": True,
            "message": "All character memories have been reset to defaults"
        })
    except Exception as e:
        print(f"Error resetting memories: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/clear_knowledge', methods=['POST'])
def clear_knowledge():
    """API endpoint to reset game knowledge to defaults"""
    try:
        # Define paths for knowledge files
        default_knowledge_path = 'data/game_knowledge_default.json'
        current_knowledge_path = 'data/game_knowledge.json'
        
        # Check if default knowledge file exists
        if os.path.exists(default_knowledge_path):
            # Read the default knowledge
            with open(default_knowledge_path, 'r') as f:
                default_knowledge = json.load(f)
            
            # Replace the current knowledge with defaults
            knowledge_engine.knowledge_base = default_knowledge
            
            # Save the reset knowledge to the current knowledge file
            with open(current_knowledge_path, 'w') as f:
                json.dump(default_knowledge, f, indent=2)
            
            return jsonify({
                "success": True, 
                "message": "Game knowledge has been reset to defaults"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Default knowledge file not found"
            }), 404
    except Exception as e:
        print(f"Error resetting knowledge: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/refresh_models', methods=['POST'])
def refresh_models():
    """API endpoint to refresh the list of available models"""
    try:
        logger.info("Refreshing available models list")
        
        # Refresh Ollama models
        ollama_manager.available_models['ollama'] = ollama_manager.get_ollama_models()
        
        # Refresh OpenAI models if applicable
        if ollama_manager.ai_provider == 'openai' and ollama_manager.openai_api_key:
            ollama_manager.available_models['openai'] = ollama_manager.get_openai_models()
        
        # Update game state
        game_state['available_models'] = ollama_manager.available_models
        
        logger.info(f"Models refreshed: {ollama_manager.available_models}")
        
        return jsonify({
            "success": True,
            "available_models": ollama_manager.available_models
        })
    except Exception as e:
        logger.error(f"Error refreshing models: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/model_status', methods=['GET'])
def check_model_status():
    """API endpoint to check the download status of tracked models"""
    try:
        # Update model status for all tracked models
        for model_name in TRACKED_MODELS:
            try:
                # Check if the model is in the available models list
                available_models = ollama_manager.get_ollama_models()
                if model_name in available_models:
                    model_status[model_name]['downloaded'] = True
                    model_status[model_name]['in_progress'] = False
                    model_status[model_name]['error'] = None
                else:
                    # If not available, check if it's being downloaded
                    # This uses a command to check Ollama's status
                    import subprocess
                    process = subprocess.Popen(
                        f"ollama list", 
                        shell=True, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        universal_newlines=True
                    )
                    stdout, stderr = process.communicate()
                    
                    # Check for "pulling" status in the output
                    if model_name in stdout and "pulling" in stdout.lower():
                        model_status[model_name]['in_progress'] = True
                        model_status[model_name]['downloaded'] = False
                    elif model_status[model_name]['in_progress']:
                        # If it was in progress before but not found in pulling status
                        # and not in available models, it might have failed
                        model_status[model_name]['in_progress'] = False
                        model_status[model_name]['error'] = "Download may have failed"
            except Exception as e:
                logger.error(f"Error checking status for model {model_name}: {e}")
                model_status[model_name]['error'] = str(e)
                
        # Return the current status of all tracked models
        return jsonify({
            "success": True,
            "model_status": model_status,
            "available_models": ollama_manager.available_models['ollama']
        })
    except Exception as e:
        logger.error(f"Error checking model status: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    character_data = prompt_engine.load_characters()
    Mem_manager.save_characters(character_data)
    Mem_manager.set_ollama(ollama_manager)
    executive_planner.set_ollama(ollama_manager)
    game_state['all_characters'] = character_data
    simulation_state['all_characters'] = character_data    
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        webbrowser.open('http://127.0.0.1:5001')
    #app.run(debug=True, port=5001)
    app.run(host='0.0.0.0', debug=True, port=5001)