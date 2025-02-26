from flask import Flask, render_template, request, jsonify

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
        
        # Simple response for testing
        if player_input.lower() == 'hello':
            response = "Greetings, traveler! Welcome to the tavern."
        elif player_input.lower() == 'look':
            response = "You see a cozy tavern with a few patrons. The tavernkeeper is wiping down the counter."
        else:
            response = f"The tavernkeeper nods at your words: '{player_input}'"
        
        return jsonify({
            "response": response,
            "game_state": game_state
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)