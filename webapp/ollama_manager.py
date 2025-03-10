import json
import os
import requests
import datetime
import prompt_engine as pe

def get_response(data, game_state, Mem_manager):
    # Save prompt to a text file
    data["prompt"] = pe.add_system_prompt(data, game_state, Mem_manager)                           

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"prompt_{timestamp}.txt"
    os.makedirs("prompts", exist_ok=True)
    with open(f"prompts/{filename}", "w", encoding="utf-8") as f:
        f.write(data["prompt"])
        
    response = requests.post(
        "http://localhost:11434/api/generate",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data)
    )
    
    # Ensure the response is JSON serializable
    response_json = response.json()
    return response_json.get('response', 'No response')

