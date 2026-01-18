from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_PATH = PROJECT_ROOT / "models"
IMAGE_MODELS_PATH = MODELS_PATH / "image"
AUDIO_MODELS_PATH = MODELS_PATH / "audio"
DEBUG_PATH = PROJECT_ROOT / "debug"