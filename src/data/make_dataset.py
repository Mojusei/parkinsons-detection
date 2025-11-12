# src/data/make_dataset.py
import pandas as pd

from config.paths import DATA_PATH
# from sklearn.model_selection import train_test_split


def load_data() -> pd.DataFrame:
    """
    Загружает датасет и удаляет неинформативный столбец 'name'.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Датасет по пути '{DATA_PATH}' не найден")
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns='name')


def split_data(
    df: pd.DataFrame,
    target_col: str = "status",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Разделяет датасет на обучающую и тестовую выборки
    с сохранением баланса классов.
    """
    pass
