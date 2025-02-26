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
