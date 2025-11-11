from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent 
DATA_PATH = ROOT_DIR / "data" / "raw" / "parkinsons.data"
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"