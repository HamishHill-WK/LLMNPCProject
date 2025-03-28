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
            
            # Create data for first response
            data = {
                "model": "deepseek-r1:8b",
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
                "model": "deepseek-r1:8b",
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
                ollama_service=ollama_manager
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
                
        #data["knowledge_analysis"] = knowledge_analysis
        
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
