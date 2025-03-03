import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

os.makedirs('model_artifacts', exist_ok=True)

print("="*50)
print("CAR PRICE PREDICTION - MODEL COMPARISON")
print("="*50)

# Load preprocessed data
print("\nLoading preprocessed data...")
try:
    with open('model_artifacts/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    print(f"Loaded data: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples")
except FileNotFoundError:
    print("Error: Preprocessed data not found. Please run the preprocessing script first.")
    exit(1)

# Load baseline metrics
try:
    with open('model_artifacts/baseline_metrics.pkl', 'rb') as f:
        baseline_metrics = pickle.load(f)
    print("Loaded baseline model metrics for comparison")
except FileNotFoundError:
    print("Warning: Baseline metrics not found. Will proceed without baseline comparison.")
    baseline_metrics = None

# Define function to evaluate regression models
def evaluate_regression_model(model, X_train, X_val, y_train, y_val, model_name="Model", verbose=False):
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Fix any negative predictions (car prices can't be negative)
    y_train_pred = np.maximum(y_train_pred, 0)
    y_val_pred = np.maximum(y_val_pred, 0)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    
    train_rmse = np.sqrt(train_mse)
    val_rmse = np.sqrt(val_mse)
    
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
    val_mape = np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100
    
    # Only print detailed metrics if verbose is True
    if verbose:
        print(f"\n{model_name} Performance:")
        print(f"{'Metric':<20} {'Training':<15} {'Validation':<15}")
        print("-" * 50)
        print(f"{'MAE':<20} ${train_mae:.2f}{'':>8} ${val_mae:.2f}{'':>8}")
        print(f"{'MSE':<20} ${train_mse:.2f}{'':>8} ${val_mse:.2f}{'':>8}")
        print(f"{'RMSE':<20} ${train_rmse:.2f}{'':>8} ${val_rmse:.2f}{'':>8}")
        print(f"{'R²':<20} {train_r2:.4f}{'':>10} {val_r2:.4f}{'':>10}")
        print(f"{'MAPE':<20} {train_mape:.2f}%{'':>9} {val_mape:.2f}%{'':>9}")
    
    return {
        'model_name': model_name,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'train_mse': train_mse,
        'val_mse': val_mse,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'train_mape': train_mape,
        'val_mape': val_mape,
        'train_predictions': y_train_pred,
        'val_predictions': y_val_pred
    }

# Define models to evaluate
models = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
}

# Train and evaluate models
print("\n" + "="*50)
print("Training and evaluating models...")
print("="*50)

results = []
if baseline_metrics:
    results.append(baseline_metrics)

trained_models = {}

# Process each model with a progress indicator
for i, (name, model) in enumerate(models.items(), 1):
    print(f"[{i}/{len(models)}] Training {name}...", end="", flush=True)
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    model_results = evaluate_regression_model(model, X_train, X_val, y_train, y_val, name, verbose=False)
    model_results['training_time'] = training_time
    results.append(model_results)
    
    # Store the trained model
    trained_models[name] = model
    
    print(f" Done in {training_time:.2f}s (RMSE: ${model_results['val_rmse']:.2f}, R²: {model_results['val_r2']:.4f})")

# Create comparison DataFrame
comparison_df = pd.DataFrame(results)
# Keep only the relevant columns for comparison
comparison_columns = ['model_name', 'val_rmse', 'val_r2', 'val_mae', 'val_mape', 'training_time']
comparison_df = comparison_df[comparison_columns].sort_values('val_rmse')

# Display comparison table
print("\n" + "="*80)
print("MODEL COMPARISON RESULTS")
print("="*80)
print(comparison_df.to_string(index=False))

# Create a single combined visualization
plt.figure(figsize=(12, 10))

# RMSE Comparison
plt.subplot(2, 1, 1)
sns.barplot(x='val_rmse', y='model_name', data=comparison_df)
plt.title('Model Comparison - RMSE (lower is better)')
plt.xlabel('Validation RMSE ($)')

# R² Comparison
plt.subplot(2, 2, 3)
sns.barplot(x='val_r2', y='model_name', data=comparison_df)
plt.title('Model Comparison - R² (higher is better)')
plt.xlabel('Validation R²')

# Performance vs. Training Time
plt.subplot(2, 2, 4)
plt.scatter(comparison_df['training_time'], comparison_df['val_rmse'], s=100, alpha=0.7)
for i, row in comparison_df.iterrows():
    plt.annotate(row['model_name'].split()[0],
                 (row['training_time'], row['val_rmse']),
                 xytext=(7, 0), 
                 textcoords='offset points',
                 ha='left')
plt.xlabel('Training Time (seconds)')
plt.ylabel('Validation RMSE ($)')
plt.title('Performance vs. Training Time')
plt.grid(True)

plt.tight_layout()
plt.savefig('model_artifacts/model_comparison_summary.png')
plt.show()

# Identify best performing model
best_model_idx = comparison_df['val_rmse'].idxmin()
best_model_row = comparison_df.loc[best_model_idx]
best_model_name = best_model_row['model_name']
best_rmse = best_model_row['val_rmse']
best_r2 = best_model_row['val_r2']

# Get the actual model object from our trained models dictionary
best_model = trained_models[best_model_name]

# Save the best model
with open(f'model_artifacts/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save all model evaluation results for future reference
with open('model_artifacts/all_model_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Save all trained models dictionary
with open('model_artifacts/all_trained_models.pkl', 'wb') as f:
    pickle.dump(trained_models, f)

print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)
print(f"Best performing model: {best_model_name}")
print(f"Validation RMSE: ${best_rmse:.2f}")
print(f"Validation R²: {best_r2:.4f}")

# Compare to baseline
if baseline_metrics:
    baseline_name = baseline_metrics['model_name']
    baseline_rmse = baseline_metrics['val_rmse']
    baseline_r2 = baseline_metrics['val_r2']
    
    rmse_improvement = (baseline_rmse - best_rmse) / baseline_rmse * 100
    r2_improvement = (best_r2 - baseline_r2) / baseline_r2 * 100
    
    print(f"\nImprovement over {baseline_name}:")
    print(f"RMSE reduction: {rmse_improvement:.2f}%")
    print(f"R² improvement: {r2_improvement:.2f}%")