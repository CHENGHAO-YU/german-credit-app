import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import pickle

class DataProcessor:
    """German Credit数据处理类"""
    
    def __init__(self):
        self.ordinal_mappings = {
            'CheckingStatus': {'no_checking': 0, 'less_0': 1, '0_to_200': 2, 'greater_200': 3},
            'CreditHistory': {'outstanding_credit': 0, 'prior_payments_delayed': 1, 'no_credits': 2, 
                            'credits_paid_to_date': 3, 'all_credits_paid_back': 4},
            'ExistingSavings': {'unknown': 0, 'less_100': 1, '100_to_500': 2, 
                              '500_to_1000': 3, 'greater_1000': 4},
            'EmploymentDuration': {'unemployed': 0, 'less_1': 1, '1_to_4': 2, 
                                 '4_to_7': 3, 'greater_7': 4},
            'Sex': {'male': 0, 'female': 1},
            'Telephone': {'none': 0, 'yes': 1},
            'ForeignWorker': {'no': 0, 'yes': 1}
        }
        
        self.onehot_cols = ['LoanPurpose', 'OthersOnLoan', 'OwnsProperty', 
                           'InstallmentPlans', 'Housing', 'Job']
        
        self.preprocessor = None
        self.feature_names = None
        self.scaled_importance = None
        
    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """预处理数据"""
        df = df.copy()
        
        # 序数编码
        for col, mapping in self.ordinal_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        # 处理目标变量
        if is_training and 'Risk' in df.columns:
            df['Risk'] = df['Risk'].map({'No Risk': 0, 'Risk': 1})
        
        # 填充缺失值
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if len(df[col].mode()) > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def get_eda_stats(self, df: pd.DataFrame) -> Dict:
        """获取EDA统计信息"""
        stats = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'describe': df.describe(include='all').to_dict()
        }
        
        if 'Risk' in df.columns:
            stats['risk_distribution'] = df['Risk'].value_counts(normalize=True).to_dict()
        
        return stats
    
    def apply_feature_scaling(self, X, feature_names, scale_factors):
        """应用特征缩放"""
        if hasattr(X, 'toarray'):
            X = X.toarray()
        df = pd.DataFrame(X, columns=feature_names)
        for i, name in enumerate(feature_names):
            df[name] *= scale_factors[i]
        return df