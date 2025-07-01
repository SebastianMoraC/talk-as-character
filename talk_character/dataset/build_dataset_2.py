"""Dataset building utilities for Rick and Morty character fine-tuning."""

import json
from typing import List, Dict, Any
from pathlib import Path
from datasets import load_dataset
from ..constants import DATA_DIR

def create_rick_conversations() -> List[Dict[str, Any]]:
    """Creates conversation pairs from Rick and Morty dataset.
    
    Returns:
        A list containing conversation pairs in chat format.
    """
    ds = load_dataset("theneuralmaze/rick-and-morty-transcripts-sharegpt")
    
    conversations = []
    for example in ds['train']:
        messages = []
        for msg in example['conversations']:
            role_map = {
                "system": "system",
                "human": "user", 
                "gpt": "assistant"
            }
            
            role = role_map.get(msg['from'])
            if role:
                messages.append({
                    "role": role,
                    "content": msg['value']
                })
        
        if len(messages) == 3:
            conversations.append({"messages": messages})
    
    return conversations

def build_dataset_2() -> None:
    """Build Rick and Morty conversation dataset."""
    Path(DATA_DIR).mkdir(exist_ok=True)
    
    conversations = create_rick_conversations()
    
    split_idx = int(len(conversations) * 0.9)
    train_examples = conversations[:split_idx]
    valid_examples = conversations[split_idx:]
    
    train_path = Path(DATA_DIR) / "train.jsonl"
    with open(train_path, 'w') as f:
        for item in train_examples:
            f.write(json.dumps(item) + '\n')
    
    valid_path = Path(DATA_DIR) / "valid.jsonl"
    with open(valid_path, 'w') as f:
        for item in valid_examples:
            f.write(json.dumps(item) + '\n')
    
    print(f"Created {len(train_examples)} training examples")
    print(f"Created {len(valid_examples)} validation examples")

__all__ = ["build_dataset_2"] 