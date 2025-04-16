from flask import Flask, render_template, request, jsonify, Response
import webbrowser
import os
import prompt_engine as pe
import json
import memory_manager
import ollama_manager as om
import re
import knowledge_engine as ke
import executive as kep  # Import the new module

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Initialize engines and managers
knowledge_engine = ke.KnowledgeEngine()
executive_planner = kep.KnowledgeExecutivePlanner(knowledge_engine=knowledge_engine)
Mem_manager = memory_manager.MemoryManager(max_short_term=3, knowledge_engine=knowledge_engine)
prompt_engine = pe.Prompt_Engine(memory_manager=Mem_manager, knowledge_engine=knowledge_engine)

# Initialize AI manager
ollama_manager = om.OllamaManager(prompt_engine=prompt_engine)

# AI service configuration
ai_config = {
    'provider': 'ollama',  # Default to ollama
    'model': 'deepseek-r1:8b',  # Default ollama model
    'openai_model': 'gpt-3.5-turbo',  # Default OpenAI model
    'available_models': {
        'ollama': ollama_manager.available_models.get('ollama', []),
        'openai': ollama_manager.available_models.get('openai', [])
    }
}

# Use the enhanced ollama_manager for all AI requests

# Simple game state
game_state = {
    'current_location': 'tavern',
    'current_npc': 'tavernkeeper',
    'all_characters': [],
    'inventory': [],
    'ai_config': ai_config
}

simulation_state = {
    'current_location': 'tavern',
    'current_speaker' : 'npc_A',
    'current_listener' : 'npc_B',
    'npc_A': 'tavernkeeper',
    'npc_B': 'blacksmith',
    'initial_prompt': 'You are in a tavern. The tavernkeeper greets you with a smile.',
    'all_characters': [],
    'inventory': [],
    'ai_config': ai_config
}

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
    print(f"app npca: {npc_a}")
    print(f"app npcb: {npc_b}")
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
            print(f"Conversation turn {i}")
            
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
            
            # Add relevant knowledge if needed
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
        
        if len(knowledge_analysis['message_types']) == 1 and ('greeting' in knowledge_analysis['message_types'] or 'farewell' in knowledge_analysis['message_types']):
            response = ollama_manager.get_response(data, game_state, Mem_manager)
            response, chain_of_thought = ollama_manager.clean_response(response)
            Mem_manager.add_interaction(game_state['current_npc'], "Player", player_input, response, chain_of_thought, game_state['current_location'])
        
        elif knowledge_analysis['knowledge_required'] or knowledge_analysis['requires_memory']:
            print("knowledge required")
            if 'memory_search_keywords' in knowledge_analysis:
                data['relevant_knowledge'] = knowledge_engine.search_knowledge_base(knowledge_analysis['memory_search_keywords'])
            response = ollama_manager.get_response(data, game_state, Mem_manager)
            response, chain_of_thought = ollama_manager.clean_response(response)
            Mem_manager.add_interaction(game_state['current_npc'], "Player", player_input, response, chain_of_thought, game_state['current_location'])
            
        if 'knowledge_query' in knowledge_analysis:
            return jsonify({
                "response": response,
                "game_state": game_state,
                "knowledge_used": knowledge_analysis.get("knowledge_required", False)
            })
        else:
            return jsonify({
                "response": response,
                "game_state": game_state
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

if __name__ == '__main__':
    character_data = prompt_engine.load_characters()
    Mem_manager.save_characters(character_data)
    Mem_manager.set_ollama(ollama_manager)
    executive_planner.set_ollama(ollama_manager)
    game_state['all_characters'] = character_data
    simulation_state['all_characters'] = character_data    
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        webbrowser.open('http://127.0.0.1:5001')
    app.run(debug=True, port=5001)