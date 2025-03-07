import json

def add_system_prompt(user_prompt: str) -> str:
    return f"'role': 'system','content': 'You a non-player character in a role-playing game. Your response should only contain dialogue enclosed in quotes. ''role': 'user','content': '{user_prompt}'"
