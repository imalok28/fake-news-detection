import os
import sys
import logging
import nltk
import torch

def setup_logger(name: str = "fake_news_detector") -> logging.Logger:
    """Configures and returns a logger for the system."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Stream Handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger

logger = setup_logger()

def setup_directories(base_dir: str):
    """Ensures that all required directories exist."""
    dirs = [
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "models", "distilbert"),
    ]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            logger.info(f"Created directory: {d}")

def download_nltk_resources(base_dir: str):
    """Downloads NLTK resources locally within the project directory."""
    nltk_data_dir = os.path.join(base_dir, "data", "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)
    
    resources = ["stopwords", "wordnet", "punkt", "omw-1.4"]
    for res in resources:
        try:
            nltk.data.find(f"corpora/{res}" if res != "punkt" else f"tokenizers/{res}")
            logger.info(f"NLTK resource '{res}' is already available.")
        except LookupError:
            logger.info(f"Downloading NLTK resource '{res}'...")
            nltk.download(res, download_dir=nltk_data_dir, quiet=True)
            logger.info(f"Downloaded NLTK resource '{res}' successfully.")

def get_torch_device() -> torch.device:
    """Determines the best available device for PyTorch (CUDA or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # Note: We avoid using MPS on macOS because PyTorch has active bugs when allocating
    # placeholder storage for dynamic-length sequences during LIME perturbations.
    else:
        device = torch.device("cpu")
    logger.info(f"Using PyTorch device: {device}")
    return device
