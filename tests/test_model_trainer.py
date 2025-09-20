import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

import pytest
import pandas as pd
import numpy as np
from model_trainer import ModelTrainer
from sklearn.datasets import make_classification

@pytest.fixture
def sample_data():
    """创建测试数据"""
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
    return X, y

@pytest.fixture
def trainer():
    """创建ModelTrainer实例"""
    return ModelTrainer()

def test_train_model(trainer, sample_data):
    """测试模型训练"""
    X, y = sample_data
    
    # 使用默认参数训练
    trainer.train_model(X, y)
    
    assert trainer.model is not None
    assert trainer.scaled_importance is not None
    assert len(trainer.scaled_importance) == X.shape[1]

def test_evaluate_model(trainer, sample_data):
    """测试模型评估"""
    X, y = sample_data
    
    # 先训练模型
    trainer.train_model(X, y)
    
    # 评估
    metrics, y_pred, y_pred_proba = trainer.evaluate_model(X, y)
    
    assert 'accuracy' in metrics
    assert 'confusion_matrix' in metrics
    assert 'classification_report' in metrics
    assert len(y_pred) == len(y)
    assert len(y_pred_proba) == len(y)
