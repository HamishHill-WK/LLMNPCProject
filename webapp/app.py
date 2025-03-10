from flask import Flask, render_template, request, jsonify, Response
import webbrowser
import os
import prompt_engine as pe
import requests
import json
import datetime
import memory_manager
import time 
import ollama_manager as om

app = Flask(__name__)
app.secret_key = 'your_secret_key'

Mem_manager = memory_manager.MemoryManager(max_short_term=10)

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
        
    return render_template('game.html', response=response, game_state=game_state)

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
    initial_prompt = request.args.get('simulation_input')
    turns = int(request.args.get('turns', 5))
    
    def generate():
        # Send SSE headers
        yield "event: start\ndata: Conversation starting\n\n"
        
        # Get initial message
        message = initial_prompt
        data = {
            "model": "deepseek-r1:7b",  # Match the Ollama model name
            "initial_prompt": initial_prompt,
            "prompt": message,
            "stream": False,
            "max_tokens" : 50
        }
        
        # Simulate conversation turns
        for i in range(turns):
            # NPC 1's turn
            npc1_response = om.get_response(data, simulation_state, Mem_manager)
            yield f"data: {json.dumps({'speaker': npc_a, 'message': npc1_response, 'turn': i * 2})}\n\n"
            #time.sleep(0.1)  # Small delay to simulate processing
            data["prompt"] = npc1_response
            # NPC 2's turn
            npc2_response = om.get_response(data, simulation_state, Mem_manager)
            yield f"data: {json.dumps({'speaker': npc_b, 'message': npc2_response, 'turn': i * 2 + 1})}\n\n"
            #time.sleep(0.1)  # Small delay to simulate processing
            
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
            "model": "deepseek-r1:7b",  # Match the Ollama model name
            "prompt": prompt,
            "stream": False,
            "max_tokens" : 50
        }
    
        # data["prompt"] = pe.add_system_prompt(data, game_state, Mem_manager)                           
        
        # # Save prompt to a text file
        # timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # filename = f"prompt_{timestamp}.txt"
        # os.makedirs("prompts", exist_ok=True)
        # with open(f"prompts/{filename}", "w", encoding="utf-8") as f:
        #     f.write(data["prompt"])
         
        # response = requests.post(
        #     "http://localhost:11434/api/generate",
        #     headers={"Content-Type": "application/json"},
        #     data=json.dumps(data)
        # )
        response = om.get_response(data, game_state, Mem_manager)
        print(response)
       # response_json = json.loads(response)
        
        Mem_manager.add_interaction(game_state['current_npc'], player_input, response, game_state['current_location'])
        
        #print(response_json)
        
        #generated_text = response_json.get('response', 'No response')

        return jsonify({
            "response": response,
            "game_state": game_state
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    game_state['all_characters'] = pe.load_characters()
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        webbrowser.open('http://127.0.0.1:5001')
    app.run(debug=True, port=5001)

