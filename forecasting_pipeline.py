import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pickle
import os

from data_preprocessing import preprocess_data

# Global variable to store training features
TRAINING_FEATURES = None

def train_models(X_train, y_train):
    """
    Train multiple regression models for sales forecasting
    
    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training target
        
    Returns:
        dict: Dictionary of trained models
    """
    global TRAINING_FEATURES
    # Store the exact feature columns used for training
    TRAINING_FEATURES = list(X_train.columns)
    
    print("\nTraining machine learning models...")
    print(f"Using {len(TRAINING_FEATURES)} features for training")
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    print("Linear Regression model trained.")
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print("Random Forest model trained.")
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    print("XGBoost model trained.")
    
    return {
        'linear_regression': lr_model,
        'random_forest': rf_model,
        'xgboost': xgb_model
    }

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate model performance using multiple metrics
    
    Args:
        y_true (Series): Actual values
        y_pred (array): Predicted values
        model_name (str): Name of the model
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R² Score: {r2:.4f}")
    
    return {
        'model_name': model_name,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }

def evaluate_all_models(models, X_test, y_test):
    """
    Evaluate all trained models
    
    Args:
        models (dict): Dictionary of trained models
        X_test (DataFrame): Testing features
        y_test (Series): Testing target
        
    Returns:
        dict: Dictionary of evaluation results for each model
    """
    results = {}
    
    for model_name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model
        results[model_name] = evaluate_model(y_test, y_pred, model_name)
    
    # Find the best model based on RMSE
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])
    print(f"\nBest model based on RMSE: {best_model[0]}")
    
    return results, best_model[0]

def plot_model_comparison(models, X_test, y_test, test_df, store_id):
    """
    Plot comparison of model predictions for a specific store
    
    Args:
        models (dict): Dictionary of trained models
        X_test (DataFrame): Testing features
        y_test (Series): Testing target
        test_df (DataFrame): Testing dataframe with dates
        store_id (int): Store ID to plot
    """
    # Filter test data for the specified store
    test_store = test_df[test_df['Store'] == store_id]
    y_test_store = y_test[test_df['Store'] == store_id]
    X_test_store = X_test[test_df['Store'] == store_id]
    
    if len(test_store) > 0:  # Make sure there's data to plot
        plt.figure(figsize=(14, 7))
        plt.plot(test_store['Date'], y_test_store, label='Actual Sales', marker='o')
        
        # Plot predictions for each model
        for model_name, model in models.items():
            y_pred = model.predict(X_test_store)
            plt.plot(test_store['Date'], y_pred, 
                    label=f'{model_name.replace("_", " ").title()} Predictions', 
                    linestyle='--', marker='x')
        
        plt.title(f'Model Comparison: Actual vs Predicted Sales for Store {store_id}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No test data available for Store {store_id}")

def align_features(df, training_features):
    """
    Align the features in the dataframe with the training features
    
    Args:
        df (DataFrame): DataFrame to align
        training_features (list): List of feature names used during training
        
    Returns:
        DataFrame: DataFrame with aligned features
    """
    # Create a new DataFrame with the same index as the input DataFrame
    aligned_df = pd.DataFrame(index=df.index)
    
    # Remove duplicate columns from the input DataFrame
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Add each training feature to the aligned DataFrame
    for feature in training_features:
        if feature in df.columns:
            # If the feature exists in the dataframe, use it
            aligned_df[feature] = df[feature].values
        else:
            # Check if this is a one-hot encoded column (e.g., StoreType_a)
            if '_' in feature:
                prefix, suffix = feature.split('_', 1)
                # Check if we have other columns with the same prefix
                similar_cols = [col for col in df.columns if col.startswith(prefix + '_')]
                
                if similar_cols:
                    print(f"Warning: Feature '{feature}' not found, but found similar columns: {similar_cols}")
                    # If we have similar columns but not this specific one, it's likely a different one-hot encoding
                    # Set this feature to 0
                    aligned_df[feature] = 0
                else:
                    print(f"Warning: Feature '{feature}' and no similar columns found. Adding with zeros.")
                    aligned_df[feature] = 0
            else:
                print(f"Warning: Feature '{feature}' not found in prediction data. Adding with zeros.")
                aligned_df[feature] = 0
    
    return aligned_df

def forecast_sales(model, historical_data, future_data, features):
    """
    Forecast sales using the trained model
    
    Args:
        model: Trained model
        historical_data (DataFrame): Historical data
        future_data (DataFrame): Future data with dates and store information
        features (list): List of feature column names
        
    Returns:
        DataFrame: DataFrame with forecasted sales
    """
    global TRAINING_FEATURES
    
    # Create a copy of future data with a placeholder for Sales
    future_data_with_sales = future_data.copy()
    future_data_with_sales['Sales'] = 0  # Placeholder
    
    # Use average customers from historical data
    future_data_with_sales['Customers'] = historical_data.groupby('Store')['Customers'].mean().mean()
    
    # Ensure PromoInterval is properly formatted
    if 'PromoInterval' in future_data_with_sales.columns:
        future_data_with_sales['PromoInterval'] = future_data_with_sales['PromoInterval'].fillna('')
        future_data_with_sales['PromoInterval'] = future_data_with_sales['PromoInterval'].astype(str)
    
    # Combine historical and future data for preprocessing
    combined_df = pd.concat([historical_data, future_data_with_sales], ignore_index=True)
    combined_df = combined_df.sort_values(['Store', 'Date'])
    
    # Ensure PromoInterval is properly formatted in combined data
    if 'PromoInterval' in combined_df.columns:
        combined_df['PromoInterval'] = combined_df['PromoInterval'].fillna('')
        combined_df['PromoInterval'] = combined_df['PromoInterval'].astype(str)
    
    # Preprocess the combined data
    processed_combined = preprocess_data(combined_df)
    
    # Remove duplicate columns if any
    processed_combined = processed_combined.loc[:, ~processed_combined.columns.duplicated()]
    
    # Extract the future data after preprocessing
    processed_future = processed_combined[processed_combined['Date'] > historical_data['Date'].max()]
    
    # Debug information
    print(f"Processed future data shape: {processed_future.shape}")
    print(f"Processed future data columns: {processed_future.columns.tolist()}")
    
    # Use the exact same features that were used during training
    if TRAINING_FEATURES is not None:
        print(f"Using {len(TRAINING_FEATURES)} features from training for prediction")
        X_future = align_features(processed_future, TRAINING_FEATURES)
    else:
        print("Warning: TRAINING_FEATURES is None. Using features provided in the function call.")
        # Check if we have all required features
        missing_features = [f for f in features if f not in processed_future.columns]
        if missing_features:
            print(f"Warning: Missing features in future data: {missing_features}")
            # Add missing features with zeros
            for feature in missing_features:
                processed_future[feature] = 0
        
        # Ensure features are in the same order as during training
        X_future = processed_future[features]
    
    # Debug information
    print(f"X_future shape: {X_future.shape}")
    
    # Make predictions
    future_predictions = model.predict(X_future)
    
    # Add predictions to the future dataframe
    processed_future['Predicted_Sales'] = future_predictions
    
    return processed_future

def save_model_and_results(model, forecast_results, model_filename='sales_forecast_model.pkl', 
                          results_filename='sales_forecast_results.csv'):
    """
    Save the trained model and forecast results
    
    Args:
        model: Trained model to save
        forecast_results (DataFrame): Forecast results to save
        model_filename (str): Filename for the model
        results_filename (str): Filename for the results
    """
    global TRAINING_FEATURES
    
    # Save the model
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_filename}")
    
    # Save the training features
    if TRAINING_FEATURES is not None:
        features_filename = model_filename.replace('.pkl', '_features.txt')
        with open(features_filename, 'w') as file:
            file.write('\n'.join(TRAINING_FEATURES))
        print(f"Training features saved to {features_filename}")
    
    # Save the forecast results
    forecast_results.to_csv(results_filename, index=False)
    print(f"Forecast results saved to {results_filename}")

def run_forecasting_pipeline(processed_df, features, forecast_days=14, store_ids=None, original_df=None):
    """
    Run the complete forecasting pipeline
    
    Args:
        processed_df (DataFrame): Processed dataframe
        features (list): List of feature column names
        forecast_days (int): Number of days to forecast
        store_ids (list): List of store IDs to forecast for
        original_df (DataFrame): Original dataframe before preprocessing (for future date generation)
        
    Returns:
        tuple: (best_model, forecast_results) - Best model and forecast results
    """
    from data_preprocessing import split_train_test, generate_future_dates
    
    print("\nRunning forecasting pipeline...")
    
    # Split data into training and testing sets
    train_df, test_df = split_train_test(processed_df, test_days=14)
    
    # Prepare features and target
    X_train = train_df[features]
    y_train = train_df['Sales']
    X_test = test_df[features]
    y_test = test_df['Sales']
    
    print(f"Training set: {len(X_train)} samples with {len(features)} features")
    print(f"Testing set: {len(X_test)} samples")
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    results, best_model_name = evaluate_all_models(models, X_test, y_test)
    
    # If store_ids is not provided, use all unique stores in the dataset
    if store_ids is None:
        store_ids = processed_df['Store'].unique()
    
    # Plot model comparison for the first store
    plot_model_comparison(models, X_test, y_test, test_df, store_ids[0])
    
    # Generate future dates for forecasting
    last_date = processed_df['Date'].max()
    
    # Use the original dataframe for future date generation if provided
    df_for_future = original_df if original_df is not None else processed_df
    
    future_df = generate_future_dates(last_date, forecast_days, store_ids, df_for_future)
    print(f"Generated future dates for forecasting: {future_df['Date'].min()} to {future_df['Date'].max()}")
    
    # Forecast sales using the best model
    best_model = models[best_model_name]
    forecast_results = forecast_sales(best_model, processed_df, future_df, features)
    print("Sales forecast completed")
    
    # Save model and results
    save_model_and_results(best_model, forecast_results, 
                          model_filename=f'rossmann_sales_forecast_{best_model_name}_model.pkl')
    
    return best_model, forecast_results

