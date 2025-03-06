from flask import Flask, render_template, request, jsonify
import webbrowser
import os
import prompt_engine as pe
import requests
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Simple game state
game_state = {
    'current_location': 'tavern',
    'current_npc': 'tavernkeeper',
    'inventory': []
}

@app.route('/', methods=['GET', 'POST'])
def game():
    response = "Welcome to the Text Adventure Game. Type something to get started."
    
    if request.method == 'POST':
        player_input = request.form.get('player_input', '')
        response = f"You said: {player_input}"
        
    return render_template('game.html', response=response, game_state=game_state)

@app.route('/api/interact', methods=['POST'])
def api_interact():
    """API endpoint for AJAX interactions"""
    try:
        data = request.json
        if not data or 'player_input' not in data:
            return jsonify({"error": "Missing player input"}), 400
            
        player_input = data['player_input']
        
        # # Simple response for testing
        # if player_input.lower() == 'hello':
        #     response = "Greetings, traveler! Welcome to the tavern."
        # elif player_input.lower() == 'look':
        #     response = "You see a cozy tavern with a few patrons. The tavernkeeper is wiping down the counter."
        # else:
        #     response = f"The tavernkeeper nods at your words: '{player_input}'"
        
        prompt = player_input.strip()
        data = {
            "model": "deepseek-r1:7b",  # Match the Ollama model name
            "prompt": prompt,
            "stream": False,
            "max_tokens" : 50
        }

        data["prompt"] = pe.add_system_prompt(data)                            
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps(data)
        )
        
        response_json = response.json()
        
        print(response_json)
        
        generated_text = response_json.get('response', 'No response')

        return jsonify({
            "response": generated_text,
            "game_state": game_state
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        webbrowser.open('http://127.0.0.1:5001')
    app.run(debug=True, port=5001)

