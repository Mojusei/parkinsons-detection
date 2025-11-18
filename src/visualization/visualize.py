# src/visualization/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import pandas as pd


def save_and_show(fig, save_path: Path = None) -> None:
    """
    Всегда отображает график. Сохраняет его, если указан save_path.

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
                "save_path должен указывать на файл"
                "с расширением (например, .png), "
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
        ) -> None:
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
        ) -> None:
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
        ) -> None:
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


def plot_confusion_matrix(y_true, y_pred, save_path: Path = None) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.ylabel("Истинный класс")
    plt.xlabel("Предсказанный класс")
    if save_path:
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_classification_report_as_heatmap(
        y_true,
        y_pred,
        save_path: Path = None
) -> None:
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).iloc[:-1, :].T

    plt.figure(figsize=(8, 4))
    sns.heatmap(
        report_df.astype(float),
        annot=True,
        cmap="viridis",
        cbar=True,
        fmt=".2f"
        )
    plt.title("Classification Report (Heatmap)")
    if save_path:
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(
        importance_dict: dict,
        top_n: int = 15,
        save_path: Path = None
        ) -> None:
    """
    Строит горизонтальный barplot топ-N самых важных признаков.

    Parameters
    ----------
    importance_dict : dict
        Словарь {признак: важность}
    top_n : int
        Сколько топ-признаков отобразить
    save_path : Path, optional
    """
    # Сортируем по убыванию
    sorted_items = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
        )
    top_features = dict(sorted_items[:top_n])

    fig, ax = plt.subplots(figsize=(8, top_n * 0.3))
    sns.barplot(
        x=list(top_features.values()),
        y=list(top_features.keys()),
        ax=ax,
        palette="viridis"
    )
    ax.set_xlabel("Важность (gain)")
    ax.set_title(f"Топ-{top_n} важных признаков")
    plt.tight_layout()
    save_and_show(fig, save_path)
