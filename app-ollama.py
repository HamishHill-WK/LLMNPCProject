# app.py - Main game application
from flask import Flask, render_template, request, session, jsonify
from npc_manager import NPCManager
import os
import json
import logging

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key')  # For session management

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("game.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize NPC manager with Ollama
try:
    logger.info("Initializing NPC Manager with Ollama integration")
    npc_manager = NPCManager()
    logger.info("NPC Manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize NPC Manager: {e}")
    raise

@app.route('/', methods=['GET', 'POST'])
def game():
    if 'game_state' not in session:
        initialize_game(session)
    
    if request.method == 'POST':
        player_input = request.form['player_input']
        response = process_player_input(player_input, session)
        return render_template('game.html', response=response, game_state=session['game_state'])
    
    return render_template('game.html', response="Welcome to the game.", game_state=session['game_state'])

@app.route('/api/interact', methods=['POST'])
def api_interact():
    """API endpoint for AJAX interactions with NPCs"""
    data = request.json
    if not data or 'player_input' not in data:
        return jsonify({"error": "Missing player input"}), 400
        
    # Get session or create a new one
    if 'game_state' not in session:
        initialize_game(session)
    
    # Process the input
    player_input = data['player_input']
    response = process_player_input(player_input, session)
    
    # Return JSON response for AJAX
    return jsonify({
        "response": response,
        "game_state": session['game_state']
    })

# Add config endpoints for model management
@app.route('/api/models', methods=['GET'])
def list_models():
    """List available models in Ollama"""
    models = npc_manager.llm_service.list_available_models()
    return jsonify({"models": models})

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get information about the current model"""
    info = npc_manager.llm_service.get_model_info()
    return jsonify(info)

def initialize_game(session):
    """Initialize a new game state"""
    session['game_state'] = {
        'current_location': 'tavern',
        'current_npc': 'tavernkeeper',
        'inventory': [],
        'quest_status': {},
        'npc_memories': {}
    }
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)

def process_player_input(player_input, session):
    """Process player input and return appropriate response"""
    # Log player input
    logger.info(f"Player input: {player_input}")
    
    # Basic command parsing (simplified)
    if player_input.lower() == 'help':
        return """Available commands:
- talk [character]: Approach a character to speak with
- say [message]: Say something to the current character
- go [location]: Travel to a different location
- look: Examine your surroundings
- inventory: Check what you're carrying
- help: Display this help message"""
    
    elif player_input.lower() == 'look':
        location_descriptions = {
            'tavern': "A cozy tavern with a roaring fireplace. The tavernkeeper is wiping down the counter while patrons chat quietly at their tables.",
            'market': "A bustling market with vendors selling all manner of goods. The merchant eyes you from behind a stall of exotic wares.",
            'blacksmith': "The heat from the forge hits you as you enter. The blacksmith hammers away at glowing metal.",
            'town_square': "The central square of the town, where people gather. The village elder sits on a bench observing passers-by.",
            'forest': "A dark forest with ancient trees. You notice a mysterious stranger watching you from behind a large oak."
        }
        
        location = session['game_state']['current_location']
        return location_descriptions.get(location, f"You are at the {location}.")
        
    elif player_input.lower() == 'inventory':
        inventory = session['game_state']['inventory']
        if not inventory:
            return "You aren't carrying anything."
        return f"You are carrying: {', '.join(inventory)}"
        
    elif player_input.startswith('talk '):
        npc_id = player_input[5:]
        if npc_id in npc_manager.characters:
            session['game_state']['current_npc'] = npc_id
            return f"You approach the {npc_manager.characters[npc_id]['name']}."
        else:
            return f"There's no one here by that name."
            
    elif player_input.startswith('say '):
        message = player_input[4:]
        response = npc_manager.get_npc_response(
            session['game_state']['current_npc'], 
            message,
            session['game_state']
        )
        return response
    
    elif player_input.startswith('go '):
        location = player_input[3:]
        # In a real game, you would check if the location is valid
        valid_locations = ['tavern', 'market', 'blacksmith', 'town_square', 'forest']
        if location in valid_locations:
            session['game_state']['current_location'] = location
            
            # Update current NPC based on location
            location_npcs = {
                'tavern': 'tavernkeeper',
                'blacksmith': 'blacksmith',
                'market': 'merchant',
                'town_square': 'village_elder',
                'forest': 'mysterious_stranger'
            }
            
            if location in location_npcs and location_npcs[location] in npc_manager.characters:
                session['game_state']['current_npc'] = location_npcs[location]
                
            return f"You arrive at the {location}."
        else:
            return f"You can't go there."