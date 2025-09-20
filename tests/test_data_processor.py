import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

import pytest
import pandas as pd
import numpy as np
from data_processor import DataProcessor

@pytest.fixture
def sample_data():
    """创建测试数据"""
    return pd.DataFrame({
        'CheckingStatus': ['no_checking', 'less_0', '0_to_200'],
        'Age': [25, 35, 45],
        'CreditAmount': [1000, 5000, 10000],
        'Risk': ['Risk', 'No Risk', 'Risk']
    })

@pytest.fixture
def processor():
    """创建DataProcessor实例"""
    return DataProcessor()

def test_preprocess_data(processor, sample_data):
    """测试数据预处理"""
    processed = processor.preprocess_data(sample_data, is_training=True)
    
    # 检查Risk列是否正确编码
    assert processed['Risk'].dtype in [np.int64, np.float64]
    assert set(processed['Risk'].unique()).issubset({0, 1})
    
    # 检查CheckingStatus是否正确编码
    assert processed['CheckingStatus'].dtype in [np.int64, np.float64]

def test_get_eda_stats(processor, sample_data):
    """测试EDA统计"""
    stats = processor.get_eda_stats(sample_data)
    
    assert 'shape' in stats
    assert 'missing_values' in stats
    assert 'dtypes' in stats
    assert stats['shape'] == (3, 4)
