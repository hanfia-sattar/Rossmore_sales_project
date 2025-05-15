import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def display_basic_statistics(df):
    """
    Display basic statistics of the dataset
    
    Args:
        df (DataFrame): Dataset to analyze
    """
    print("\nBasic Statistics:")
    print(df[['Sales', 'Customers', 'Promo', 'SchoolHoliday']].describe())
    
    # Count of stores
    print(f"\nNumber of unique stores: {df['Store'].nunique()}")
    
    # Date range
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Missing values
    print("\nMissing values:")
    print(df.isnull().sum())

def plot_sales_by_store(df, store_ids=None, max_stores=5):
    """
    Plot sales distribution by store
    
    Args:
        df (DataFrame): Dataset to analyze
        store_ids (list): List of store IDs to plot, if None, uses first max_stores
        max_stores (int): Maximum number of stores to plot if store_ids is None
    """
    if store_ids is None:
        store_ids = df['Store'].unique()[:max_stores]
    
    df_subset = df[df['Store'].isin(store_ids)]
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Store', y='Sales', data=df_subset)
    plt.title('Sales Distribution by Store')
    plt.xlabel('Store')
    plt.ylabel('Sales')
    plt.grid(True, alpha=0.3, axis='y')
    plt.show()

def plot_sales_by_day_of_week(df, store_ids=None, max_stores=5):
    """
    Plot sales distribution by day of week
    
    Args:
        df (DataFrame): Dataset to analyze
        store_ids (list): List of store IDs to plot, if None, uses first max_stores
        max_stores (int): Maximum number of stores to plot if store_ids is None
    """
    if store_ids is None:
        store_ids = df['Store'].unique()[:max_stores]
    
    df_subset = df[df['Store'].isin(store_ids)]
    
    # Map day of week to day names
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_mapping = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 
                   5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
    df_subset['DayName'] = df_subset['DayOfWeek'].map(day_mapping)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='DayName', y='Sales', data=df_subset, order=day_names)
    plt.title('Sales Distribution by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Sales')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.show()

def plot_sales_by_store_type(df):
    """
    Plot sales distribution by store type
    
    Args:
        df (DataFrame): Dataset to analyze
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='StoreType', y='Sales', data=df)
    plt.title('Sales Distribution by Store Type')
    plt.xlabel('Store Type')
    plt.ylabel('Sales')
    plt.grid(True, alpha=0.3, axis='y')
    plt.show()

def plot_sales_by_month(df, store_ids=None, max_stores=5):
    """
    Plot sales distribution by month
    
    Args:
        df (DataFrame): Dataset to analyze
        store_ids (list): List of store IDs to plot, if None, uses first max_stores
        max_stores (int): Maximum number of stores to plot if store_ids is None
    """
    if store_ids is None:
        store_ids = df['Store'].unique()[:max_stores]
    
    df_subset = df[df['Store'].isin(store_ids)]
    
    # Extract month and year
    df_subset['Month'] = df_subset['Date'].dt.month
    df_subset['Year'] = df_subset['Date'].dt.year
    
    # Group by month and calculate average sales
    monthly_sales = df_subset.groupby(['Year', 'Month'])['Sales'].mean().reset_index()
    monthly_sales['YearMonth'] = monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str).str.zfill(2)
    
    plt.figure(figsize=(14, 6))
    plt.plot(monthly_sales['YearMonth'], monthly_sales['Sales'], marker='o')
    plt.title('Average Sales by Month')
    plt.xlabel('Year-Month')
    plt.ylabel('Average Sales')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_sales_vs_promo(df):
    """
    Plot sales distribution by promotion status
    
    Args:
        df (DataFrame): Dataset to analyze
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Promo', y='Sales', data=df)
    plt.title('Sales Distribution by Promotion Status')
    plt.xlabel('Promotion (0=No, 1=Yes)')
    plt.ylabel('Sales')
    plt.grid(True, alpha=0.3, axis='y')
    plt.show()

def plot_correlation_matrix(df):
    """
    Plot correlation matrix of numerical features
    
    Args:
        df (DataFrame): Dataset to analyze
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr = numeric_df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.show()

def run_exploratory_analysis(df, store_ids=None, max_stores=5):
    """
    Run a complete exploratory data analysis
    
    Args:
        df (DataFrame): Dataset to analyze
        store_ids (list): List of store IDs to analyze, if None, uses first max_stores
        max_stores (int): Maximum number of stores to analyze if store_ids is None
    """
    print("\nPerforming exploratory data analysis...")
    
    # Display basic statistics
    display_basic_statistics(df)
    
    # Plot sales by store
    plot_sales_by_store(df, store_ids, max_stores)
    
    # Plot sales by day of week
    plot_sales_by_day_of_week(df, store_ids, max_stores)
    
    # Plot sales by store type
    plot_sales_by_store_type(df)
    
    # Plot sales by month
    plot_sales_by_month(df, store_ids, max_stores)
    
    # Plot sales vs promotion
    plot_sales_vs_promo(df)
    
    # Plot correlation matrix
    plot_correlation_matrix(df)
    
    print("Exploratory data analysis completed.")

