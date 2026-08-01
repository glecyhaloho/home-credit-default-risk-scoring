from pathlib import Path

RAW_DIR = Path.home() / "Documents" / "Home_Credit_Rakamin" / "home-credit-default-risk"
PROJECT_DIR = Path.home() / "Documents" / "Home_Credit_Rakamin" / "home-credit-scoring"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
