"""Dataset building utilities for character fine-tuning."""

import json
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from ..constants import DATA_FRAME_FILE, DATA_DIR, TRAIN_JSONL, SYSTEM_PROMPT

JESSE_SYSTEM_PROMPT = """You are Jesse Pinkman from Breaking Bad. You are a young former student turned methamphetamine manufacturer and dealer.
- Street-smart but not academically intelligent
Stay completely in character as Jesse Pinkman. Speak like a young street-smart guy from Albuquerque."""

def load_breaking_bad_dataset() -> pd.DataFrame:
    """Loads the Breaking Bad transcript dataset from CSV.
    
    This function loads the Breaking Bad transcript dataset from csv file.
    
    Returns:
        pandas.DataFrame: A dataframe containing Breaking Bad episode transcripts.
            The dataframe includes columns for actor, text, season, episode, and title.
    """
    df = pd.read_csv(DATA_FRAME_FILE)
    # Remove rows with null values in actor or text columns
    df = df.dropna(subset=['actor', 'text'])
    return df

def create_conversation_pairs(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Creates conversation pairs from the Breaking Bad transcript dataset.
    
    This function processes the dataframe to create conversation pairs where a non-Jesse character
    speaks followed by Jesse's response. Uses chat format with system prompt defining Jesse's character.
    
    Args:
        df: The Breaking Bad transcript dataframe containing dialogue and speaker information.
    
    Returns:
        A list containing conversation pairs in chat format:
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": {non_jesse_dialogue}},
                    {"role": "assistant", "content": {jesse_dialogue}}
                ]
            }
    """
    new_rows = []
    for i in tqdm(range(len(df) - 1), desc="Creating conversation pairs"):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]
        if current_row["actor"] != "Jesse" and next_row["actor"] == "Jesse":
            if (current_row["season"] == next_row["season"] and 
                current_row["episode"] == next_row["episode"]):
                
                # Only keep examples with substantial Jesse responses (50+ chars)
                jesse_response = next_row['text']
                if len(jesse_response) >= 50:
                    # Format for MLX chat training
                    new_rows.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": current_row['text']},
                            {"role": "assistant", "content": jesse_response}
                        ]
                    })
    return new_rows

def build_dataset() -> None:
    """Build Jesse Pinkman conversation dataset from Breaking Bad transcripts."""
    # Create walter_data directory
    Path(DATA_DIR).mkdir(exist_ok=True)
    
    df = load_breaking_bad_dataset()
    conversations = create_conversation_pairs(df)
    
    # Split into train/valid (90/10 split as required by MLX)
    split_idx = int(len(conversations) * 0.9)
    train_examples = conversations[:split_idx]
    valid_examples = conversations[split_idx:]
    
    # Write train.jsonl
    with open(TRAIN_JSONL, 'w') as f:
        for item in train_examples:
            f.write(json.dumps(item) + '\n')
    
    # Write valid.jsonl (required by MLX)
    valid_path = Path(DATA_DIR) / "valid.jsonl"
    with open(valid_path, 'w') as f:
        for item in valid_examples:
            f.write(json.dumps(item) + '\n')
    
    print(f"Created {len(train_examples)} training examples")
    print(f"Created {len(valid_examples)} validation examples")

__all__ = [
    "load_breaking_bad_dataset",
    "create_conversation_pairs", 
    "build_dataset"
] 