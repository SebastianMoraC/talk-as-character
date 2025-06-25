"""talk-character: A clean, reusable package for character-based LLM fine-tuning with MLX."""

__version__ = "0.1.0"

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

__all__ = ["__version__"] 