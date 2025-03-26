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

knowledge_engine = ke.KnowledgeEngine()
executive_planner = kep.KnowledgeExecutivePlanner(knowledge_engine=knowledge_engine)
Mem_manager = memory_manager.MemoryManager(max_short_term=3, knowledge_engine=knowledge_engine)
prompt_engine = pe.Prompt_Engine(memory_manager=Mem_manager, knowledge_engine=knowledge_engine)
ollama_manager = om.OllamaManager(prompt_engine=prompt_engine)

# Simple game state
game_state = {
    'current_location': 'tavern',
    'current_npc': 'tavernkeeper',
    'all_characters': [],
    'inventory': []
}

simulation_state = {
    'current_location': 'tavern',
    'current_speaker' : 'npc_A',
    'current_listener' : 'npc_B',
    'npc_A': 'tavernkeeper',
    'npc_B': 'blacksmith',
    'initial_prompt': 'You are in a tavern. The tavernkeeper greets you with a smile.',
    'all_characters': [],
    'inventory': []
}

@app.route('/', methods=['GET', 'POST'])
def game():
    response = "Welcome to the Text Adventure Game. Type something to get started."
    
    if request.method == 'POST':
        player_input = request.form.get('player_input', '')
        response = f"You said: {player_input}"
        
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
        
        print(f"app sim npca: {simulation_state['npc_A']}")
        print(f"app sim npcb: {simulation_state['npc_B']}")
        
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
        data = {
            "model": "deepseek-r1:7b",  # Match the Ollama model name
            "initial_prompt": message,
            "prompt": message,
            "stream": False,
            "max_tokens" : 50
        }
                
        # Simulate conversation turns
        for i in range(turns):
            print("appL boops")
            data["prompt"] = message

            npc1_response = ollama_manager.get_response(data, simulation_state, Mem_manager)

            # Switch speaker in simulation state
            Mem_manager.add_interaction(simulation_state[simulation_state['current_speaker']], simulation_state['current_listener'], message, npc1_response, simulation_state['current_location'])
            simulation_state['current_listener'] = simulation_state['current_speaker']
            simulation_state['current_speaker'] = 'npc_A' if simulation_state['current_speaker'] == 'npc_B' else 'npc_B'
            npc1_response = re.sub(r'<think>.*?</think>', '', npc1_response, flags=re.DOTALL).strip()
            yield f"data: {json.dumps({'speaker': simulation_state[simulation_state['current_speaker']], 'message': npc1_response, 'turn': i * 2})}\n\n"
            data["prompt"] = npc1_response
            # NPC 2's turn
            npc2_response = ollama_manager.get_response(data, simulation_state, Mem_manager)
            Mem_manager.add_interaction(simulation_state[simulation_state['current_speaker']], simulation_state[simulation_state['current_listener']], npc1_response, npc2_response, simulation_state['current_location'])
            npc2_response = re.sub(r'<think>.*?</think>', '', npc2_response, flags=re.DOTALL).strip()
            simulation_state['current_listener'] = simulation_state['current_speaker']
            simulation_state['current_speaker'] = 'npc_A' if simulation_state['current_speaker'] == 'npc_B' else 'npc_B'
            yield f"data: {json.dumps({'speaker': simulation_state[simulation_state['current_speaker']], 'message': npc2_response, 'turn': i * 2 + 1})}\n\n"
            
            # Update message for next turn
            message = npc2_response
            
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
            "model": "deepseek-r1:8b",  # Match the Ollama model name
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
        
        data["knowledge_analysis"] = knowledge_analysis
        
        knowledge_engine.assess_knowledge(
            player_input=player_input,
            character_id=game_state['current_npc'],
            conversation_context=conversation_context,
            data_dict=data,
            ollama_service=ollama_manager
        )
        
        response = ""
        # If knowledge is required, retrieve it
        if knowledge_analysis.get("knowledge_required", False) or knowledge_analysis.get("requires_memory", False):
            print(f"Knowledge required for: {knowledge_analysis.get('knowledge_query', '')}")
        
        if len(knowledge_analysis['message_types']) == 1 and ('greeting' in knowledge_analysis['message_types'] or 'farewell' in knowledge_analysis['message_types']):
            response = ollama_manager.get_response(data, game_state, Mem_manager)
            response, chain_of_thought = ollama_manager.clean_response(response)
            Mem_manager.add_interaction(game_state['current_npc'], "Player", player_input, response, chain_of_thought, game_state['current_location'])
        
        elif knowledge_analysis['knowledge_required'] or knowledge_analysis['requires_memory']:
            print("knowledge required")
            # Get response from LLM
            response = ollama_manager.get_response(data, game_state, Mem_manager)
            response, chain_of_thought = ollama_manager.clean_response(response)
            Mem_manager.add_interaction(game_state['current_npc'], "Player", player_input, response, chain_of_thought, game_state['current_location'])
            
        return jsonify({
            "response": response,
            "game_state": game_state,
            "knowledge_used": knowledge_analysis.get("knowledge_required", False)
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

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
