# Data Directory

## Overview
This directory contains banking and loan data used for bad loan detection model training and data quality auditing.

## Dataset Information

### Data Source
- **Source**: Bank loan portfolio datasets
- **Type**: Loan records with borrower and performance information
- **Records**: Varies by bank data size
- **Features**: Borrower demographics, loan details, performance metrics

### Data Structure
```
data/
├── raw/
│   ├── loan_portfolio.csv
│   ├── borrower_profile.csv
│   └── loan_performance.csv
├── processed/
│   ├── cleaned_loans.csv
│   ├── engineered_features.csv
│   └── train_test_data/
└── quality/
    └── audit_reports.json
```

## Key Data Features
- **Loan Features**: Amount, Interest Rate, Term, Purpose, Grade
- **Borrower Features**: Income, Employment, Credit Score, DTI Ratio
- **Performance Features**: Default Status, Payment History, Delinquency Days
- **Temporal**: Issue Date, Maturity Date, Last Payment Date

## Target Variable
- **Bad Loan**: Binary classification (0 = Current/Paid, 1 = Default/Charged-Off)
- **Bad Loan Rate**: Percentage of loans in default state

## Data Quality Metrics
The `DataQualityAudit` class evaluates:
- Missing values count and percentage
- Duplicate records identification
- Outlier detection and handling
- Data consistency checks
- Statistical summaries and reports

## Data Preprocessing
1. Handle missing values: Imputation or removal
2. Feature engineering: Ratio calculations, time-based features
3. Encoding: Categorical to numerical conversion
4. Scaling: Standardization for ML models
5. Class balancing: Handle imbalanced loan defaults

## Usage
The `BadLoanDetector` class loads and preprocesses data from this directory.
The `DataQualityAudit` class generates quality reports.

## Data Privacy
All sensitive information is anonymized. Real names, SSNs, and addresses are excluded from datasets.
