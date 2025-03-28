import os
import json
from datetime import datetime

class WorkingMemory:
    """Manages the current conversation history."""
    
    def __init__(self, chat_history_dir):
        self.chat_history_dir = chat_history_dir
        os.makedirs(chat_history_dir, exist_ok=True)
        self.messages = []
        self.conversation_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def initialize_with_system_message(self, system_message):
        """Initialize conversation with a system message if messages list is empty"""
        if not self.messages:
            self.messages = [{"role": "system", "content": system_message}]
    
    def add_user_message(self, message):
        """Add a user message to the conversation history"""
        self.messages.append({"role": "user", "content": message})
        
    def add_assistant_message(self, message):
        """Add an assistant message to the conversation history"""
        self.messages.append({"role": "assistant", "content": message})
    
    def get_messages(self):
        """Return the full conversation history"""
        return self.messages
    
    def save_conversation(self):
        """Save the current conversation to a file named with the start time"""
        if not self.messages:
            return None
            
        filename = f"{self.conversation_start_time}.json"
        filepath = os.path.join(self.chat_history_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.messages, f, indent=2)
            
        return filepath
