# src/data/make_dataset.py
import pandas as pd
from typing import Tuple
from config.paths import DATA_DIR
from sklearn.model_selection import train_test_split


def load_data() -> pd.DataFrame:
    """
    Загружает датасет из CSV-файла и удаляет столбец 'name'.

    Returns
    -------
    pd.DataFrame
        Датафрейм без столбца 'name'.

    Raises
    ------
    FileNotFoundError
        Если файл по указанному пути не существует.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Датасет по пути '{DATA_DIR}' не найден")
    df = pd.read_csv(DATA_DIR)
    return df.drop(columns='name')


def split_data(
    df: pd.DataFrame,
    target_col: str = "status",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Разделяет датасет на обучающую и тестовую выборки
    с сохранением баланса классов.

    Parameters
    ----------
    df : pd.DataFrame
        Исходный датафрейм с признаками и целевой переменной.
    target_col : str, optional
        Название столбца с целевой переменной (по умолчанию "status").
    test_size : float, optional
        Доля тестовой выборки (по умолчанию 0.2).
    random_state : int, optional
        Фиксированный seed для воспроизводимости (по умолчанию 42).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)

    Raises
    ------
    ValueError
        Если нужный столбец отсутствует в датафрейме.
    """
    if target_col not in df.columns:
        raise ValueError(f"Колонка '{target_col}' не найдена в датафрейме")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
