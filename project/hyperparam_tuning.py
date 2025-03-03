import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

print("="*50)
print("CAR PRICE PREDICTION - HYPERPARAMETER TUNING")
print("="*50)

# Load preprocessed data
print("\nLoading preprocessed data...")
with open('model_artifacts/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_val = data['X_val']
y_train = data['y_train']
y_val = data['y_val']

# Load the best model
print("\nLoading best model from comparison phase")
with open('model_artifacts/best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Evaluate original model performance
y_val_pred = best_model.predict(X_val)
original_val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
original_val_r2 = r2_score(y_val, y_val_pred)

print(f"\nOriginal Model Performance:")
print(f"Validation RMSE: ${original_val_rmse:.2f}")
print(f"Validation R²: {original_val_r2:.4f}")

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Initialize base model
rf = RandomForestRegressor(random_state=42)

# Set up GridSearchCV
print("\nStarting hyperparameter tuning with GridSearchCV")
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Fit GridSearchCV
start_time = time.time()
grid_search.fit(X_train, y_train)
tuning_time = time.time() - start_time
print(f"\nHyperparameter tuning completed in {tuning_time:.2f} seconds")

# Get best parameters
print("\nBest parameters:", grid_search.best_params_)
best_cv_score = -grid_search.best_score_
print(f"Best cross-validation RMSE: ${best_cv_score:.2f}")

# Get the best model
best_tuned_model = grid_search.best_estimator_

# Evaluate on validation set
y_val_pred_tuned = best_tuned_model.predict(X_val)
tuned_val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_tuned))
tuned_val_r2 = r2_score(y_val, y_val_pred_tuned)

print("\nTuned Model Performance:")
print(f"Validation RMSE: ${tuned_val_rmse:.2f}")
print(f"Validation R²: {tuned_val_r2:.4f}")

# Calculate improvement
val_improvement = (original_val_rmse - tuned_val_rmse) / original_val_rmse * 100
print(f"\nImprovement: {val_improvement:.2f}% RMSE reduction")

# Save the tuned model
with open('model_artifacts/tuned_rf_model.pkl', 'wb') as f:
    pickle.dump(best_tuned_model, f)

print("\nTuned model saved to 'model_artifacts/tuned_rf_model.pkl'")

feature_count = X_train.shape[1]  # Get number of features
print(f"\nSaving feature count: {feature_count}")
with open('model_artifacts/feature_count.pkl', 'wb') as f:
    pickle.dump(feature_count, f)