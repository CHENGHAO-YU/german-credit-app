import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import optuna
import pickle
import json

class ModelTrainer:
    """模型训练类"""
    
    def __init__(self):
        self.model = None
        self.best_params = None
        self.preprocessor = None
        self.feature_names = None
        self.scaled_importance = None
        
    def create_preprocessor(self, onehot_cols):
        """创建预处理器"""
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols)
            ],
            remainder='passthrough'
        )
        return self.preprocessor
    
    def optimize_hyperparameters(self, X_train, y_train, n_trials=30):
        """使用Optuna优化超参数"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'verbose': -1
            }
            
            model = LGBMClassifier(**params)
            return cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
        
        study = optuna.create_study(direction='maximize', study_name='lgbm_optimization')
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        return study
    
    def train_model(self, X_train, y_train):
        """训练模型"""
        if self.best_params is None:
            # 使用默认参数
            self.best_params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1
            }
        
        self.model = LGBMClassifier(**self.best_params)
        self.model.fit(X_train, y_train)
        
        # 计算特征重要性
        importances = self.model.feature_importances_
        self.scaled_importance = importances / (importances.max() + 1e-8)
        
        return self.model
    
    def evaluate_model(self, X_val, y_val):
        """评估模型"""
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist(),
            'classification_report': classification_report(y_val, y_pred, output_dict=True)
        }
        
        return metrics, y_pred, y_pred_proba
    
    def save_model(self, filepath='models/model.pkl'):
        """保存模型"""
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'best_params': self.best_params,
            'feature_names': self.feature_names,
            'scaled_importance': self.scaled_importance
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath='models/model.pkl'):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.best_params = model_data['best_params']
        self.feature_names = model_data['feature_names']
        self.scaled_importance = model_data['scaled_importance']
        
        return self