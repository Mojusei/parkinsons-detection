# src/models/train_model.py
from src.data.make_dataset import load_data, split_data
from src.models.train_pipeline import (
    train_and_evaluate,
    cross_validate_model,
    get_feature_importance
)
from src.visualization.visualize import (
    plot_confusion_matrix,
    plot_classification_report_heatmap,
    plot_feature_importance
)
from src.paths import ROOT_DIR, DATA_PATH, MODEL_PATH, FIGURES_DIR
from sklearn.metrics import classification_report
import joblib
import pandas as pd


def main() -> None:
    """
    Основная функция обучения модели с полной отчётностью.
    Вызывается через CLI: `poetry run train`.

    Выполняет:
    - Кросс-валидацию на полных данных
    - Обучение на 80% данных
    - Вывод classification report в консоль
    - Сохранение модели, важности признаков и графиков
    """

    # Загрузка данных
    df = load_data(DATA_PATH)
    X = df.drop(columns=["status"])
    y = df["status"]

    # Кросс-валидация
    cv_mean, cv_std = cross_validate_model(X, y, cv_folds=5)
    print(f"🔍 Кросс-валидация (5-fold): {cv_mean:.2%} ± {cv_std:.2%}")

    # Разделение и обучение
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline, test_acc, y_pred = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test
        )

    # Classification report (текстовый)
    print("\n📋 Classification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Здоров", "Болен"]
        ))

    # Сохранение модели
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n✅ Модель сохранена: {MODEL_PATH}")
    print(f"🎯 Точность на тесте: {test_acc:.2%}")

    # Feature Importance
    feature_names = X.columns.tolist()
    importance = get_feature_importance(pipeline, feature_names)

    # Сохранение важности в CSV
    importance_df = pd.DataFrame(
        sorted(importance.items(), key=lambda x: x[1], reverse=True),
        columns=["Признак", "Важность"]
    )
    importance_df.to_csv(
        ROOT_DIR / "reports" / "feature_importance.csv",
        index=False
        )
    print("Важность признаков сохранена: reports/feature_importance.csv")

    # Графики
    plot_confusion_matrix(
        y_test,
        y_pred,
        FIGURES_DIR / "confusion_matrix.png"
        )
    plot_classification_report_heatmap(
        y_test,
        y_pred,
        FIGURES_DIR / "classification_report.png"
        )
    plot_feature_importance(
        importance,
        top_n=15,
        save_path=FIGURES_DIR / "feature_importance.png"
        )


if __name__ == '__main__':
    main()
