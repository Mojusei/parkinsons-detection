from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data" / "parkinsons.data"
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
