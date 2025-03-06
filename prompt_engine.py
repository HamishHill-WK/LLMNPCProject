import json

def add_system_prompt(user_prompt: str) -> str:
    return f"'role': 'system','content': 'You a non-player character in a role-playing game. the player approaches you and says : ''role': 'user','content': '{user_prompt}'"
