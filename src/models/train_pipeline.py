# src/models/train_pipeline.py
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold


def create_pipeline() -> Pipeline:
    """
    Создаёт ML-пайплайн с масштабированием признаков и XGBoost-классификатором.

    Returns
    -------
    Pipeline
        Пайплайн из StandardScaler и XGBClassifier.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        ))
    ])


def train_and_evaluate(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
        ) -> Tuple[Pipeline, float, np.ndarray]:
    """
    Обучает пайплайн и оценивает его на тестовой выборке.

    Parameters
    ----------
    X_train : pd.DataFrame
        Признаки обучающей выборки.
    X_test : pd.DataFrame
        Признаки тестовой выборки.
    y_train : pd.Series
        Целевая переменная обучающей выборки.
    y_test : pd.Series
        Целевая переменная тестовой выборки.

    Returns
    -------
    Tuple[Pipeline, float, np.ndarray]
        (обученный pipeline, точность на тесте, предсказания на тесте)
    """
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return pipeline, accuracy, y_pred


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5
) -> Tuple[float, float]:
    """
    Выполняет стратифицированную кросс-валидацию и
    возвращает среднюю точность и стандартное отклонение.

    Parameters
    ----------
    X : pd.DataFrame
        Признаки (все данные).
    y : pd.Series
        Целевая переменная.
    cv_folds : int, optional
        Число фолдов для кросс-валидации (по умолчанию 5).

    Returns
    -------
    Tuple[float, float]
        (средняя точность, стандартное отклонение)
    """
    pipeline = create_pipeline()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    return scores.mean(), scores.std()


def get_feature_importance(
    pipeline: Pipeline,
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Извлекает важность признаков из обученной XGBoost-модели в пайплайне.

    Parameters
    ----------
    pipeline : Pipeline
        Обученный пайплайн, содержащий XGBClassifier в шаге 'model'.
    feature_names : List[str]
        Список названий признаков (в том же порядке, что и в X).

    Returns
    -------
    Dict[str, float]
        Словарь вида {название_признака: важность}.
    """
    importance = pipeline.named_steps["model"].feature_importances_
    return dict(zip(feature_names, importance))
