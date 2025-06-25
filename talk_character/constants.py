"""Configuration constants for talk-character package."""

from typing import Dict, List, Any

# Paths
DATA_DIR = "walter_data"
RAW_CSV = f"{DATA_DIR}/breaking_bad_transcripts.csv"
TRAIN_JSONL = f"{DATA_DIR}/train.jsonl"

# Legacy paths for backward compatibility
DATA_FRAME_FILE = "BB_data.csv"

# Model + LoRA
MODEL_CONFIG: Dict[str, Any] = {
    "model_name": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",  # Back to accessible model
    "quantize": True,      # QLoRA works fine for demos
    "dtype": "float16",
}

LORA_CONFIG: Dict[str, Any] = {
    "r": 16,               # Slightly higher rank for better learning
    "alpha": 32,           # 2x rank as recommended  
    "dropout": 0.1,        # Small dropout for regularization
    "num_layers": 16,      # Standard value from docs
    "modules": ["self_attn.q_proj", "self_attn.v_proj"],  # Standard attention modules
}

# Training
TRAINING_ARGS: Dict[str, Any] = {
    "batch_size": 1,       # Keep low for memory
    "epochs": 3,           # Fewer epochs
    "lr": 1e-5,            # Keep learning rate
    "warmup_steps": 20,    # Minimal warmup
    "seed": 42,
    "output_dir": "walter_checkpoints",
    "save_steps": 100,     # Save less frequently
    "eval_steps": 100,     # Evaluate less frequently  
    "max_steps": 600,      # More training for better results
}

MAX_SEQ_LENGTH = 2048

SYSTEM_PROMPT = (
    "You are Jesse Pinkman from Breaking Bad – a young, street-smart "
    "methamphetamine manufacturer. Stay completely in character and speak "
    "like a guy from Albuquerque."
)

__all__ = [
    "DATA_DIR", "RAW_CSV", "TRAIN_JSONL",
    "DATA_FRAME_FILE",
    "MODEL_CONFIG", "LORA_CONFIG", "TRAINING_ARGS",
    "MAX_SEQ_LENGTH", "SYSTEM_PROMPT"
]

