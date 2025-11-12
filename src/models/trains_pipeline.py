# src/models/train_pipeline.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


def create_pipeline():
    """
    Создаёт ML-пайплайн: StandardScaler + XGBClassifier.
    """
    
    


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Обучает пайплайн и возвращает его вместе с точностью на тесте.
    """
    
