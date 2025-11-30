"""Automated Bad Loan Detection System

This module contains the machine learning model and data quality audit functions
for detecting bad loans in banking datasets.

Author: Data Science Team
Date: November 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import joblib
from datetime import datetime


class DataQualityAudit:
    """Perform comprehensive data quality checks on loan datasets."""
    
    def __init__(self, df):
        self.df = df
        self.audit_report = {}
    
    def check_missing_values(self):
        """Identify missing values in dataset."""
        missing = self.df.isnull().sum()
        self.audit_report['missing_values'] = missing[missing > 0].to_dict()
        return self.audit_report['missing_values']
    
    def check_outliers(self, column, threshold=3):
        """Detect outliers using z-score method."""
        z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
        outliers = (z_scores > threshold).sum()
        return outliers
    
    def check_duplicates(self):
        """Find duplicate records."""
        duplicates = self.df.duplicated().sum()
        self.audit_report['duplicates'] = duplicates
        return duplicates
    
    def generate_report(self):
        """Generate complete audit report."""
        self.check_missing_values()
        self.check_duplicates()
        self.audit_report['timestamp'] = datetime.now().isoformat()
        self.audit_report['total_records'] = len(self.df)
        self.audit_report['total_columns'] = len(self.df.columns)
        return self.audit_report


class BadLoanDetector:
    """Machine learning model for detecting bad loans."""
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = None
    
    def build_model(self):
        """Build the ML model based on selected type."""
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42
            )
        return self.model
    
    def preprocess_data(self, X, y=None, fit=True):
        """Preprocess and encode data."""
        X_processed = X.copy()
        
        # Handle categorical variables
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if fit:
                self.encoders[col] = LabelEncoder()
                X_processed[col] = self.encoders[col].fit_transform(X_processed[col].astype(str))
            else:
                X_processed[col] = self.encoders[col].transform(X_processed[col].astype(str))
        
        # Scale numeric features
        if fit:
            X_processed = self.scaler.fit_transform(X_processed)
        else:
            X_processed = self.scaler.transform(X_processed)
        
        return X_processed
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model on provided data."""
        self.feature_names = X_train.columns
        X_train_processed = self.preprocess_data(X_train, fit=True)
        
        if X_val is not None:
            X_val_processed = self.preprocess_data(X_val, fit=False)
            self.model.fit(X_train_processed, y_train, 
                          eval_set=[(X_val_processed, y_val)], verbose=0)
        else:
            self.model.fit(X_train_processed, y_train)
    
    def predict(self, X):
        """Make predictions on new data."""
        X_processed = self.preprocess_data(X, fit=False)
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)
        return predictions, probabilities
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            return feature_importance
        return None
    
    def save_model(self, filepath):
        """Save trained model to file."""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load pre-trained model from file."""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("Bad Loan Detection System initialized")
    print("Models available: xgboost, lightgbm, random_forest")
