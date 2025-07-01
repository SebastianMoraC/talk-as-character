"""MLX Trainer for character fine-tuning."""

import subprocess
import sys
from pathlib import Path
from ..constants import MODEL_CONFIG, LORA_CONFIG, TRAINING_ARGS

class MlxTrainer:
    """MLX-based trainer for character fine-tuning."""
    
    def train(self, output_dir: str = "walter_checkpoints/walter_run") -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", MODEL_CONFIG["model_name"],
            "--train",
            "--data", "walter_data",
            "--num-layers", str(LORA_CONFIG["num_layers"]), 
            "--iters", str(TRAINING_ARGS["max_steps"]),
            "--batch-size", str(TRAINING_ARGS["batch_size"]),
            "--learning-rate", str(TRAINING_ARGS["lr"]),
            "--steps-per-report", "10",
            "--steps-per-eval", "50",
            "--save-every", str(TRAINING_ARGS["save_steps"]),
            "--adapter-path", output_dir,
            "--mask-prompt"
        ]
        
        subprocess.run(cmd, check=True)

__all__ = ["MlxTrainer"] 