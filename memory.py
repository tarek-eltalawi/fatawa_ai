from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Message:
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class ConversationMemory:
    def __init__(self, max_messages: int = 10):
        self.messages: List[Message] = []
        self.max_messages = max_messages
    
    def add_message(self, role: str, content: str) -> None:
        """Add a new message to the conversation history."""
        message = Message(role=role, content=content)
        self.messages.append(message)
        
        # Trim history if it exceeds max_messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_conversation_history(self) -> str:
        """Get formatted conversation history for context."""
        history = []
        for msg in self.messages:
            history.append(f"{msg.role.capitalize()}: {msg.content}")
        return "\n".join(history)
    
    def clear(self) -> None:
        """Clear the conversation history."""
        self.messages = []
    
    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert conversation history to a list of dictionaries."""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in self.messages
        ] 