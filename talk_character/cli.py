"""Main CLI interface for talk-character package."""

import click
import shutil
import subprocess
import sys
from pathlib import Path
from .constants import MODEL_CONFIG, TRAINING_ARGS

@click.group()
@click.version_option()
def cli():
	"""talk-character: Character-based LLM fine-tuning with MLX."""
	pass

@cli.command(name="clean")
def clean_cmd():
	"""Clean all generated data folders."""
	folders_to_remove = ["walter_data", "walter_checkpoints", "walter_fused_model"]

	for folder in folders_to_remove:
		folder_path = Path(folder)
		if folder_path.exists():
			shutil.rmtree(folder_path)
			print(f"Removed {folder}")
		else:
			print(f"Skipped {folder} (not found)")

@cli.command(name="build-dataset")
def build_dataset_cmd():
	"""Build Jesse Pinkman conversation dataset from Breaking Bad transcripts."""
	from .dataset.build_dataset import build_dataset
	build_dataset()

@cli.command(name="build-dataset-2")
def build_dataset_2_cmd():
	"""Build Rick and Morty conversation dataset from Hugging Face."""
	from .dataset.build_dataset_2 import build_dataset_2
	build_dataset_2()

@cli.command(name="finetune")
def finetune_cmd():
	"""Fine-tune Jesse Pinkman character model with MLX."""
	from .llm.finetune import finetune
	finetune()

@cli.command(name="export")
def export_cmd():
	"""Export model adapters and create Ollama instructions."""
	from .llm.finetune import export
	export()

@cli.command(name="serve")
def serve_cmd():
	"""Start HTTP server to chat with Jesse Pinkman model."""
	adapter_path = Path(f"{TRAINING_ARGS['output_dir']}/walter_run")
	
	if adapter_path.exists():
		print(f"🔥 Starting server with base model + Jesse adapters...")
		subprocess.run([
			sys.executable, "-m", "mlx_lm", "server", 
			"--model", MODEL_CONFIG["model_name"],
			"--adapter-path", str(adapter_path),
			"--port", "8080"
		], check=True)
	else:
		print("❌ No adapters found. Run 'finetune' first.")
		sys.exit(1)

if __name__ == "__main__":
	cli() 
