from typing import Dict, List, Any

DATA_DIR = "walter_data"
RAW_CSV = f"{DATA_DIR}/breaking_bad_transcripts.csv"
TRAIN_JSONL = f"{DATA_DIR}/train.jsonl"

DATA_FRAME_FILE = "BB_data.csv"

MODEL_CONFIG: Dict[str, Any] = {
	"model_name": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
	"quantize": True,
	"dtype": "float16",
}

LORA_CONFIG: Dict[str, Any] = {
	"r": 16,
	"alpha": 32,
	"dropout": 0.1,
	"num_layers": 16,
	"modules": ["self_attn.q_proj", "self_attn.v_proj"],
}

TRAINING_ARGS: Dict[str, Any] = {
	"batch_size": 1,
	"epochs": 3,
	"lr": 1e-5,
	"warmup_steps": 20,
	"seed": 42,
	"output_dir": "walter_checkpoints",
	"save_steps": 100,
	"eval_steps": 100,
	"max_steps": 600,
}

MAX_SEQ_LENGTH = 2048

SYSTEM_PROMPT = (
	"You are Jesse Pinkman from Breaking Bad – a young, street-smart "
	"methamphetamine manufacturer. Stay completely in character and speak "
	"like a guy from Albuquerque."
)
SYSTEM_PROMPT_RICK = ("You are an interdimensional genius scientist named Rick Sanchez.\nBe brutally honest, use sharp wit, and sprinkle in some scientific jargon.\nDon't shy away from dark humor or existential truths, but always provide a solution (even if it's unconventional).")

__all__ = [
	"DATA_DIR", "RAW_CSV", "TRAIN_JSONL",
	"DATA_FRAME_FILE",
	"MODEL_CONFIG", "LORA_CONFIG", "TRAINING_ARGS",
	"MAX_SEQ_LENGTH", "SYSTEM_PROMPT", "SYSTEM_PROMPT_RICK"
]

