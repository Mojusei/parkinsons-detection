# src/models/train_model.py
from src.data.make_dataset import load_data, split_data
from src.models.train_pipeline import (
    train_and_evaluate,
    cross_validate_model,
    get_feature_importance
)
from src.models.io import save_model, save_feature_importance
from src.visualization.visualize import (
    plot_confusion_matrix,
    plot_classification_report_heatmap,
    plot_feature_importance
)
from sklearn.metrics import classification_report


def main() -> None:
    """
    Основной CLI-интерфейс для обучения модели.
    Запускается командой: poetry run train
    """
    # Загрузка данных
    df = load_data()

    X = df.drop(columns=["status"])
    y = df["status"]

    # Кросс-валидация
    cv_mean, cv_std = cross_validate_model(X, y, cv_folds=5)
    print(f"Кросс-валидация (5-fold): {cv_mean:.2%} ± {cv_std:.2%}")

    # Обучение
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline, test_acc, y_pred = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test
        )

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Здоров", "Болен"]
        ))

    # Сохранение модели и важности
    model_path = save_model(pipeline, model_name="parkinsons_xgb.joblib")
    importance = get_feature_importance(pipeline, X.columns.tolist())
    importance_path = save_feature_importance(
        importance,
        filename="parkinsons_feature_importance.csv"
        )

    print(f"\nМодель сохранена: {model_path}")
    print(f"Важность признаков: {importance_path}")
    print(f"Точность на тесте: {test_acc:.2%}")

    # Сохранение графиков
    plot_confusion_matrix(y_test, y_pred, show=False, save=True)
    plot_classification_report_heatmap(y_test, y_pred, show=False, save=True)
    plot_feature_importance(importance, top_n=15, show=False, save=True)

    print("Все графики сохранены в reports/figures/")


if __name__ == '__main__':
    main()
