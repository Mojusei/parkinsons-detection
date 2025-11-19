# src/visualization/visualize.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from sklearn.metrics import confusion_matrix, classification_report

from config.paths import FIGURES_DIR


# ================
# БАЗОВЫЕ УТИЛИТЫ
# ================

def show_figure(fig: plt.Figure) -> None:
    """
    Отображает указанную matplotlib-фигуру, гарантируя, что она активна.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Фигура для отображения.
    """
    plt.figure(fig.number)
    plt.show()


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """
    Сохраняет фигуру в файл в директории FIGURES_DIR.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Фигура для сохранения.
    filename : str
        Имя файла с расширением (например, "plot.png").

    Returns
    -------
    Path
        Полный путь к сохранённому файлу.

    Raises
    ------
    ValueError
        Если filename не содержит расширения.
    """
    if Path(filename).suffix == "":
        raise ValueError(
            "Имя файла должно содержать расширение (например, .png),"
            f" получено: '{filename}'"
        )
    save_path = FIGURES_DIR / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _show_and_save_if_needed(
    fig: plt.Figure,
    filename: str,
    show: bool,
    save: bool
) -> Optional[Path]:
    """
    Вспомогательная функция для оркестрации показа и сохранения фигуры.

    Parameters
    ----------
    fig : plt.Figure
        Фигура для обработки.
    filename : str
        Имя файла для сохранения (если save=True).
    show : bool
        Нужно ли показать фигуру.
    save : bool
        Нужно ли сохранить фигуру.

    Returns
    -------
    Path or None
        Путь к файлу, если save=True, иначе None.
    """
    if show:
        show_figure(fig)
    if save:
        return save_figure(fig, filename)
    return None


# ======================
# ФУНКЦИИ СОЗДАНИЯ ГРАФИКОВ
# ======================

def _create_class_distribution_plot(
    df: pd.DataFrame,
    target_col: str = "status"
) -> Tuple[plt.Figure, plt.Axes]:
    """Создаёт фигуру распределения классов."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(
        data=df,
        x=target_col,
        ax=ax,
        hue=target_col,
        palette="Set2",
        legend=False
        )
    ax.set_title("Распределение классов")
    ax.set_xlabel("Диагноз (0 = здоров, 1 = болен)")
    return fig, ax


def _create_feature_distributions_plot(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = "status"
) -> Tuple[plt.Figure, plt.Axes]:
    """Создаёт boxplot распределения признаков по классам."""
    df_melted = df.melt(
        id_vars=[target_col],
        value_vars=features,
        var_name="Признак",
        value_name="Значение"
        )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df_melted,
        x="Признак",
        y="Значение",
        hue=target_col,
        palette="Set2",
        ax=ax
    )
    ax.set_title("Распределение ключевых признаков по классам")
    ax.tick_params(axis='x', rotation=45)
    return fig, ax


def _create_target_correlation_plot(
    df: pd.DataFrame,
    target_col: str = "status"
) -> Tuple[plt.Figure, plt.Axes]:
    """Создаёт heatmap корреляции с целевой переменной."""
    if target_col not in df.columns:
        raise ValueError(f"Cтолбец '{target_col}' отсутствует в данных.")

    corr_with_target = df.corr()[target_col].drop(target_col)
    corr_df = corr_with_target.to_frame(name="Корреляция с " + target_col)
    corr_df = corr_df.sort_values(
        by=corr_df.columns[0],
        key=abs,
        ascending=False
        )

    fig, ax = plt.subplots(figsize=(4, len(corr_df) * 0.3 + 1))
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
    return fig, ax


def _create_confusion_matrix_plot(
    y_true, y_pred
) -> Tuple[plt.Figure, plt.Axes]:
    """Создаёт heatmap матрицы ошибок."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax
    )
    ax.set_title("Матрица ошибок")
    ax.set_ylabel("Истинный класс")
    ax.set_xlabel("Предсказанный класс")
    return fig, ax


def _create_classification_report_heatmap_plot(
    y_true, y_pred
) -> Tuple[plt.Figure, plt.Axes]:
    """Создаёт heatmap classification report."""
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).iloc[:-1, :].T

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        report_df.astype(float),
        annot=True,
        cmap="viridis",
        cbar=True,
        fmt=".2f",
        ax=ax
    )
    ax.set_title("Classification Report (Heatmap)")
    return fig, ax


def _create_feature_importance_plot(
    importance_dict: Dict[str, float],
    top_n: int = 15
) -> Tuple[plt.Figure, plt.Axes]:
    """Создаёт barplot важности признаков."""
    sorted_items = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )
    top_features = dict(sorted_items[:top_n])

    features = list(top_features.keys())
    importances = list(top_features.values())

    fig, ax = plt.subplots(figsize=(8, top_n * 0.3))
    sns.barplot(
        x=importances,
        y=features,
        hue=features,
        palette="viridis",
        ax=ax,
        legend=False
    )
    ax.set_xlabel("Важность (gain)")
    ax.set_title(f"Топ-{top_n} важных признаков")
    plt.tight_layout()
    return fig, ax


# ======================
# ОРКЕСТРАТОРЫ
# ======================

def plot_class_distribution(
    df: pd.DataFrame,
    target_col: str = "status",
    *,
    show: bool = True,
    save: bool = True
) -> Optional[Path]:
    fig, _ = _create_class_distribution_plot(df, target_col)
    return _show_and_save_if_needed(
        fig,
        "01_class_distribution.png",
        show,
        save
        )


def plot_feature_distributions(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = "status",
    *,
    show: bool = True,
    save: bool = True
) -> Optional[Path]:
    fig, _ = _create_feature_distributions_plot(df, features, target_col)
    return _show_and_save_if_needed(
        fig,
        "02_feature_distributions.png",
        show,
        save
        )


def plot_target_correlation(
    df: pd.DataFrame,
    target_col: str = "status",
    *,
    show: bool = True,
    save: bool = True
) -> Optional[Path]:
    fig, _ = _create_target_correlation_plot(df, target_col)
    return _show_and_save_if_needed(
        fig,
        "03_target_correlation.png",
        show,
        save
        )


def plot_confusion_matrix(
    y_true,
    y_pred,
    *,
    show: bool = True,
    save: bool = True
) -> Optional[Path]:
    fig, _ = _create_confusion_matrix_plot(y_true, y_pred)
    return _show_and_save_if_needed(
        fig,
        "04_confusion_matrix.png",
        show,
        save
        )


def plot_classification_report_heatmap(
    y_true,
    y_pred,
    *,
    show: bool = True,
    save: bool = True
) -> Optional[Path]:
    fig, _ = _create_classification_report_heatmap_plot(y_true, y_pred)
    return _show_and_save_if_needed(
        fig,
        "05_classification_report.png",
        show,
        save
        )


def plot_feature_importance(
    importance_dict: Dict[str, float],
    top_n: int = 15,
    *,
    show: bool = True,
    save: bool = True
) -> Optional[Path]:
    fig, _ = _create_feature_importance_plot(importance_dict, top_n)
    return _show_and_save_if_needed(
        fig,
        "06_feature_importance.png",
        show,
        save
        )
