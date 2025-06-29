"""Fine-tuning and export for character models with MLX."""

import subprocess
import sys
from pathlib import Path
from .trainer import MlxTrainer
from ..constants import TRAINING_ARGS, MODEL_CONFIG, SYSTEM_PROMPT

def finetune() -> None:
    """Fine-tune Jesse Pinkman character model with MLX."""
    output_dir = f"{TRAINING_ARGS['output_dir']}/walter_run"
    trainer = MlxTrainer()
    trainer.train(output_dir=output_dir)

def export() -> None:
    """Export fused Jesse Pinkman model."""
    output_dir = f"{TRAINING_ARGS['output_dir']}/walter_run"
    fused_dir = "walter_fused_model"
    
    # Fuse LoRA adapters with base model
    subprocess.run([
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", MODEL_CONFIG["model_name"],
        "--adapter-path", output_dir,
        "--save-path", fused_dir
    ], check=True)
    
    # Create usage instructions
    instructions = f"""# Jesse Pinkman Model Ready!

## Test your fused model:
```bash
python -m mlx_lm generate \\
    --model {Path(fused_dir).absolute()} \\
    --prompt "Hey Jesse, what's cooking?" \\
    --max-tokens 100
```

## Chat with Jesse:
```bash
python -m mlx_lm chat \\
    --model {Path(fused_dir).absolute()}
```

Your fused Jesse Pinkman model: {Path(fused_dir).absolute()}
"""
        
    with open("JESSE_MODEL_READY.md", "w") as f:
            f.write(instructions)
        
__all__ = ["finetune", "export"] 