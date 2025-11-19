# src/models/io.py
from pathlib import Path
from typing import Dict, Any
import joblib
import pandas as pd

# Импорт путей из конфигурации
from config.paths import MODELS_DIR, REPORTS_DIR


def save_model(model: Any, model_name: str = "model.joblib") -> Path:
    """
    Сохраняет модель в файл с использованием joblib.

    Parameters
    ----------
    model : Any
        Обученная модель (Pipeline, XGBClassifier и т.д.).
    model_name : str, optional
        Имя файла модели (по умолчанию "model.joblib").

    Returns
    -------
    Path
        Полный путь к сохранённому файлу модели.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / model_name
    joblib.dump(model, model_path)
    return model_path


def load_model(model_name: str = "model.joblib") -> Any:
    """
    Загружает модель из файла.

    Parameters
    ----------
    model_name : str, optional
        Имя файла модели (по умолчанию "model.joblib").

    Returns
    -------
    Any
        Загруженная модель.

    Raises
    ------
    FileNotFoundError
        Если файл модели не найден.
    """
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    return joblib.load(model_path)


def save_feature_importance(
    importance_dict: Dict[str, float],
    filename: str = "feature_importance.csv"
) -> Path:
    """
    Сохраняет важность признаков в CSV-файл.

    Parameters
    ----------
    importance_dict : Dict[str, float]
        Словарь вида {название_признака: важность}.
    filename : str, optional
        Имя файла (по умолчанию "feature_importance.csv").

    Returns
    -------
    Path
        Полный путь к сохранённому CSV-файлу.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = REPORTS_DIR / filename

    df = pd.DataFrame(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True),
        columns=["feature", "importance"]
    )
    df.to_csv(filepath, index=False)
    return filepath
