# src/visualization/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import pandas as pd


def save_and_show(fig, save_path: Path = None):
    """
    Вспомогательная функция: сохраняет или показывает график.

    Parameters
    ----------
    fig
        Объект matplotlib.figure.Figure.
    save_path : Path, optional
        Путь для сохранения изображения (включая имя файла и расширение).
    """
    plt.show()

    if save_path is not None:
        if save_path.suffix == "":
            raise ValueError(
                f"save_path должен указывать на файл"
                f"с расширением (например, .png), "
                f"но получена папка: {save_path}"
            )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_class_balance(
        df: pd.DataFrame,
        target_col: str = "status",
        size: tuple = (6, 4),
        save_path: Path = None
        ):
    """
    График распределения классов (столбчатая диаграмма).

    Parameters
    ----------
    df : pd.DataFrame
        Датафрейм с признаками и целевой переменной.
    target_col : str, optional
        Название столбца с целевой переменной (по умолчанию "status").
    size : tuple
        Размер графика.
    save_path : Path, optional
        Путь для сохранения изображения (включая имя файла и расширение).
    """
    fig, ax = plt.subplots(figsize=size)
    sns.countplot(data=df, x=target_col, ax=ax, hue=target_col, palette="Set2")
    ax.set_title("Распределение классов")
    ax.set_xlabel("Диагноз (0 = здоров, 1 = болен)")
    save_and_show(fig, save_path)


def plot_target_correlation(
        df: pd.DataFrame,
        target_col: str = "status",
        size: tuple = (4, 8),
        save_path: Path = None
        ):
    """
    Строит тепловую карту корреляции всех признаков с целевой переменной.

    Parameters
    ----------
    df : pd.DataFrame
        Датафрейм с признаками и целевой переменной.
    target_col : str, optional
        Название столбца с целевой переменной (по умолчанию "status").
    size : tuple
        Размер графика.
    save_path : Path, optional
        Путь для сохранения изображения (включая имя файла и расширение).
    """
    if target_col not in df.columns:
        raise ValueError(f"Целевой столбец '{target_col}' "
                         f"отсутствует в данных.")

    corr_with_target = df.corr()[target_col].drop(target_col)

    corr_df = corr_with_target.to_frame(name="Корреляция с " + target_col)

    corr_df = corr_df.sort_values(
        by=corr_df.columns[0],
        key=abs,
        ascending=False
        )

    fig, ax = plt.subplots(figsize=size)

    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": "Коэффициент корреляции Пирсона"},
        ax=ax
    )

    ax.set_title(f"Корреляция признаков с '{target_col}'", fontsize=12, pad=15)
    plt.tight_layout()
    save_and_show(fig, save_path)


def plot_correlation_heatmap(
        df: pd.DataFrame,
        size: tuple = (12, 10),
        save_path: Path = None
        ):
    """
    Строит тепловую карту корреляции всех признаков с целевой переменной.

    Parameters
    ----------
    df : pd.DataFrame
        Датафрейм с признаками и целевой переменной.
    size : tuple
        Размер графика.
    save_path : Path, optional
        Путь для сохранения изображения (включая имя файла и расширение).
    """
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=size)
    sns.heatmap(corr, mask=mask, cmap="coolwarm", ax=ax)

    ax.set_title("Корреляции")
    save_and_show(fig, save_path)


def plot_confusion_matrix(y_true, y_pred, save_path: Path = None):
    pass


def plot_classification_report_as_heatmap(
        y_true,
        y_pred,
        save_path: Path = None
):
    pass
