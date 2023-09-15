import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

import warnings
warnings.filterwarnings('ignore')

# Load the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

"""
# Look into missing data
missing_values = train.isnull().sum()
missing_values_filtered = missing_values[missing_values != 0]
print(missing_values_filtered)
missing_values = test.isnull().sum()
missing_values_filtered = missing_values[missing_values != 0]
print(missing_values_filtered)
# Some of these missing values should be reported as 0 or none, as they are
# likely missing simply because they describe an absent attribute of the
# house (i.e. basement, garage, pool, etc).
"""

# Store the test Ids because they need to be dropped before training but still
# needed for the submission file
test_ids = test['Id']

# Drop utilities and street because nearly all observations are the same value.
# Drop PoolQC because nearly all the data is missing.
train = train.drop(['Id', 'Utilities', 'Street', 'PoolQC'], axis = 1)
test = test.drop(['Id', 'Utilities', 'Street', 'PoolQC'], axis = 1)

# Drop SalePrice from training and set features equal to the columns.
# SalePrice already absent in test file to prevent cheating.
train_features = train.drop(['SalePrice'], axis=1)
test_features = test
features = pd.concat([train_features, test_features]).reset_index(drop=True)

# Select numeric columns by data type, then drop MSSubClass, MoSold, and YrSold.
numeric_cols = features.select_dtypes(exclude=['object']).columns
numeric_cols = numeric_cols.difference(['MSSubClass', 'MoSold', 'YrSold'])

# Assume that the missing values here are because the house does not have a garage,
# basement, etc. Replace them with 0.
for col in ('MasVnrArea', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MiscVal'):
    features[col] = features[col].fillna(0)

# Due to the skewness of the data, replace the rest of the missing numeric
# data with the median instead of the mean.
for col in numeric_cols:
    features[col].fillna(features[col].median(), inplace=True)

"""
# Visualize numeric distributions
for col in numeric_cols:
    plt.figure(figsize=(8, 4))  # Adjust the figure size as needed
    sns.histplot(features[col])
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
"""

# Get target variable. The SalePrice data is highly skewed, so we will
# log-transform to normalize, reduce variance, and reduce impact of outliers.
train["SalePrice"] = np.log1p(train["SalePrice"])
y = train['SalePrice'].reset_index(drop=True)

# We will likewise apply a log transform to the rest of the numeric data for
# similar reasons as the target variable.
for col in numeric_cols:
    if not np.all(np.isfinite(features[col])):  # Check for non-finite values
        features[col] = np.log1p(features[col] - features[col].min() + 1)

"""
# Visualize numeric distributions after log tranformation
for col in numeric_cols:
    plt.figure(figsize=(8, 4))  # Adjust the figure size as needed
    sns.histplot(features[col])
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
"""

# Select categorical columns by data type, then add MSSubClass, MoSold, and
# YrSold. These are categorical variables represented in the data as numbers.
categorical_cols = features.select_dtypes(include=['object']).columns
categorical_cols = categorical_cols.union(['MSSubClass', 'MoSold', 'YrSold'])

# These features are often missing if the house does not have a garage,
# basement, etc. We replace missing values with "None".
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'FireplaceQu', 'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
            'BsmtFinType1', 'BsmtFinType2', 'Fence', 'MiscFeature',
            'MasVnrType']:
    features[col] = features[col].fillna('None')

# It does not make sense to replace the other missing values with "None", so we
# replace them with the mode.
for col in categorical_cols:
    features[col].fillna(features[col].mode()[0], inplace=True)

# Create dummy variables for categorical features
features = pd.get_dummies(features, columns=categorical_cols)

"""
This is an example of how I went about optimizing hyperparamters.
Some parameters I played around with manually rather than doing a
computationally expensive search.
# Split data into train and validation sets for hyperparameter tuning
X_train, X_val, y_train, y_val = train_test_split(features[:len(y)], y, test_size=0.2, random_state=42)

# Initialize models
ridge = Ridge()
lasso = Lasso()
gbr = GradientBoostingRegressor()
xgboost = XGBRegressor()
lightgbm = LGBMRegressor()

# Create a parameter grid to iterate through
param_grid = {
    'ridge': {'alpha': np.logspace(-2, 1, 4)},
    'lasso': {'alpha': np.logspace(-4, -1, 4)},
    'gbr': {'learning_rate': np.logspace(-3, 0, 3), 'n_estimators': np.logspace(2, 3, 3),
            'max_depth': [3, 4], 'min_samples_leaf': np.linspace(10, 15, 3)},
    'xgboost': {'learning_rate': np.logspace(-3, 0, 3), 'n_estimators': np.logspace(2, 3, 3),
                'max_depth': [3, 4], 'objective': 'reg:linear',
                'min_child_weight': [0, 1], 'reg_alpha': np.logspace(-2, -5, 3)},
    'lightgbm': {'learning_rate': np.logspace(-3, 0, 3), 'n_estimators': np.logspace(2, 3, 3),
                 'num_leaves': np.linspace(20, 30, 3),
                 'max_bin': np.linspace(50, 100, 4), 'min_data_in_leaf': np.linspace(4, 6, 3),
                 'bagging_frac': np.linspace(0.5, 0.8, 3), 'baggin_freq': np.linspace(4, 6, 3),
                 'feature_fraction': np.linspace(0.2, 0.5, 3)}
}

best_models = {}
for model_name, model in zip(['ridge', 'lasso', 'gbr', 'xgboost', 'lightgbm'], [ridge, lasso, gbr, xgboost, lightgbm]):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid[model_name], scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model

print(best_models)
"""

# Omptimized hyperparameters
ridge = Ridge(alpha=15.7)
# Use Lasso to reduce impact of less important features
lasso = Lasso(alpha=0.0005)
# Use huber loss function as it is robust to outliers given data skewness.
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10,
                                loss='huber')
# reg_alpha to incorporate Lasso in the ensemble
xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3500, max_depth=3,
                       subsample=0.7, colsample_bytree=0.7,
                       objective='reg:linear', scale_pos_weight=1,
                       reg_alpha=0.00005)
# Also played around with bagging fraction/freq and feature fraction
lightgbm = LGBMRegressor(objective='regression', num_leaves=5,
                         learning_rate=0.05, n_estimators=5000, max_bin = 55,
                         bagging_fraction = 0.8, bagging_freq = 5,
                         feature_fraction = 0.2, min_data_in_leaf=6,
                         verbose=-1)

# Fit the models
ridge_model_full_data = ridge.fit(features[:len(y)], y)
lasso_model_full_data = lasso.fit(features[:len(y)], y)
gbr_model_full_data = gbr.fit(features[:len(y)], y)
xgb_model_full_data = xgboost.fit(features[:len(y)], y)
lgb_model_full_data = lightgbm.fit(features[:len(y)], y)

# Make predictions
ridge_pred = ridge_model_full_data.predict(features[len(y):])
lasso_pred = lasso_model_full_data.predict(features[len(y):])
gbr_pred = gbr_model_full_data.predict(features[len(y):])
xgb_pred = xgb_model_full_data.predict(features[len(y):])
lgb_pred = lgb_model_full_data.predict(features[len(y):])

# Average predictions together
final_predictions = (ridge_pred + lasso_pred + gbr_pred + xgb_pred + lgb_pred)/5

# Reverse the log transformation on predictions
final_predictions = np.expm1(final_predictions)

# Create submission DataFrame
submission = pd.DataFrame()
submission['Id'] = test_ids
submission['SalePrice'] = final_predictions
print(submission.head())

# Create submission file
submission.to_csv('submission.csv', index=False)