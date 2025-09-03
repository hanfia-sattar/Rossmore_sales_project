import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_preprocessing import load_rossmann_data, preprocess_data, get_feature_columns
from exploratory_analysis import run_exploratory_analysis
from forecasting_pipeline import run_forecasting_pipeline
from business_insights import generate_business_insights

def main():
    """
    Main function to run the complete sales forecasting system
    """
    print("Sales Forecasting System with Kaggle Data")
    print("========================================")
    print("This application demonstrates a complete sales forecasting pipeline using Kaggle's Rossmann Store Sales dataset.")
    
    # Step 1: Load the Kaggle dataset
    print("\nStep 1: Loading and preparing Kaggle data...")
    df = load_rossmann_data()
    
    if df is None:
        print("Exiting due to data loading error.")
        exit()
    
    # Select a subset of stores for demonstration
    store_ids = df['Store'].unique()[:5]  # First 5 stores
    df_subset = df[df['Store'].isin(store_ids)]
    
    # Step 2: Perform exploratory data analysis
    print("\nStep 2: Performing exploratory data analysis...")
    run_exploratory_analysis(df_subset, store_ids)
    
    # Step 3: Preprocess data and engineer features
    print("\nStep 3: Preprocessing data and engineering features...")
    processed_df = preprocess_data(df_subset)
    print(f"Processed data shape: {processed_df.shape}")
    print("Added features:", [col for col in processed_df.columns if col not in df.columns])
    
    # Step 4: Get feature columns for modeling
    features = get_feature_columns(processed_df)
    print("Features used for modeling:", features)
    
    # Save the feature list for future reference
    with open('model_features.txt', 'w') as f:
        f.write('\n'.join(features))
    print("Feature list saved to model_features.txt")
    
    # Step 5: Run forecasting pipeline
    print("\nStep 5: Running forecasting pipeline...")
    # Pass the original dataframe for future date generation
    best_model, forecast_results = run_forecasting_pipeline(processed_df, features, 
                                                           forecast_days=14, store_ids=store_ids,
                                                           original_df=df_subset)
    
    # Step 6: Generate business insights
    print("\nStep 6: Generating business insights...")
    insights = generate_business_insights(df_subset, forecast_results)
    
    print("\nSales Forecasting System with Kaggle data completed successfully!")

if __name__ == "__main__":
    main()

