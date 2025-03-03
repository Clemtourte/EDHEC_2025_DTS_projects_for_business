import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

print("Loading preprocessed data...")
# Load preprocessed data
try:
    with open('model_artifacts/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    print(f"Loaded training data with {X_train.shape[0]} samples and {X_train.shape[1]} features")
    print(f"Loaded validation data with {X_val.shape[0]} samples")
    print(f"Loaded test data with {X_test.shape[0]} samples")
except FileNotFoundError:
    print("Error: Preprocessed data not found. Please run the preprocessing script first.")
    exit(1)

# Define function to evaluate regression models
def evaluate_regression_model(model, X_train, X_val, y_train, y_val, model_name="Model"):
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
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
    
    # Print metrics
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
        'val_mape': val_mape
    }

# Visualize predictions vs actual values
def plot_predictions(y_true, y_pred, title="Predictions vs Actual Values"):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot residuals
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Price ($)")
    plt.ylabel("Residuals ($)")
    plt.title(f"Residuals for {title}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot histogram of residuals
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel("Residual Value ($)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Residuals")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("\n" + "="*50)
print("Defining and evaluating Linear Regression as baseline model")
print("="*50)

print("\nTraining Linear Regression model")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_results = evaluate_regression_model(lr_model, X_train, X_val, y_train, y_val, "Linear Regression")

# Visualize Linear Regression predictions
y_val_pred_lr = lr_model.predict(X_val)
plot_predictions(y_val, y_val_pred_lr, "Linear Regression: Predictions vs Actual")

# Feature importance for Linear Regression
try:
    with open('model_artifacts/column_info.pkl', 'rb') as f:
        column_info = pickle.load(f)
    
    try:
        with open('model_artifacts/feature_selector.pkl', 'rb') as f:
            feature_selector = pickle.load(f)
        selected_indices = feature_selector.get_support(indices=True)
    except:
        print("Feature selector not available as a pickle file")
    
    try:
        with open('model_artifacts/preprocessing_pipeline.pkl', 'rb') as f:
            preprocessing_pipeline = pickle.load(f)
    except:
        print("Preprocessing pipeline not available as a pickle file")
    
    # Since we don't have direct access to feature names after preprocessing,
    # we'll look at the coefficients and their magnitudes
    coefficients = pd.DataFrame({
        'Feature': [f"Feature_{i}" for i in range(X_train.shape[1])],
        'Coefficient': lr_model.coef_
    })
    
    # Sort by absolute coefficient value
    coefficients['Abs_Coefficient'] = np.abs(coefficients['Coefficient'])
    coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)
    
    # Display top coefficients
    print("\nTop 10 most influential features in Linear Regression model:")
    print(coefficients.head(10))
    
    # Plot top coefficients
    plt.figure(figsize=(12, 8))
    top_n = 15
    top_coef = coefficients.head(top_n)
    
    # Create a horizontal bar chart
    sns.barplot(x='Abs_Coefficient', y='Feature', data=top_coef)
    plt.title(f'Top {top_n} Features by Importance in Linear Regression')
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Could not analyze feature importance: {e}")

# Save the model
print("\nSaving baseline model")
os.makedirs('model_artifacts', exist_ok=True)

with open('model_artifacts/linear_regression_baseline.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

# Save baseline performance metrics
with open('model_artifacts/baseline_metrics.pkl', 'wb') as f:
    pickle.dump(lr_results, f)

print("Baseline model saved successfully!")