# Parkinson's Disease Detection

Раннее предсказание болезни Паркинсона с использованием XGBoost на датасете UCI ML Parkinsons.


## Структура проекта
```text
parkinsons-detection/
├── data/raw/                 # Сырые данные (внешний источник)
├── notebooks/                # EDA и отчёт по модели
├── src/
│   ├── config/paths.py       # Централизованные пути
│   ├── data/                 # Загрузка и разделение данных
│   ├── models/               # Обучение, сохранение, оценка
│   └── visualization/        # Визуализация метрик и важности признаков
├── models/                   # Сохранённые модели (joblib)
├── reports/
│   ├── feature_importance.csv
│   └── figures/              # Графики (confusion matrix, отчёты и др.)
├── pyproject.toml            # Зависимости и CLI-команды
└── README.md
```

## Установка

1. Установите [Poetry]
2. Датасет находится по пути `data/raw/parkinsons.data`.
3. Установите зависимости:

```bash
poetry install
```


## Использование 
### Обучение модели 
```bash
poetry run train
```
Скрипт выполнит: 
- Кросс-валидацию (5-fold)
- Обучение XGBoost на 80% данных
- Сохранение модели (`models/parkinsons_xgb.joblib`)
- Генерацию отчётов: метрики, важность признаков, графики

### Анализ и визуализация
     
Откройте Jupyter-ноутбуки: 
- `notebooks/01-eda-parkinsons.ipynb` — разведочный анализ
- `notebooks/02-model-training-and-report.ipynb` — обучение


## Требования 

- Python ≥ 3.9
- Датасет: `parkinsons.data` (UCI ML Repository)


## Результаты 

- Точность на тестовой выборке: >90%
- Поддержка кросс-валидации, feature importance, classification report
- Полная воспроизводимость через CLI
     