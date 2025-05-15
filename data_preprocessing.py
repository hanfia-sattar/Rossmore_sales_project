import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

def load_rossmann_data():
    """
    Load and prepare the Rossmann Store Sales dataset from Kaggle.
    
    Returns:
        DataFrame: Prepared sales data
    """
    print("Loading Rossmann Store Sales data from Kaggle...")
    
    try:
        # Load the sales data
        sales_data = pd.read_csv('Data/train.csv')
        
        # Load the store data
        store_data = pd.read_csv('Data/store.csv')
        
        print(f"Successfully loaded sales data with {len(sales_data)} rows and {sales_data.shape[1]} columns")
        print(f"Successfully loaded store data with {len(store_data)} rows and {store_data.shape[1]} columns")
        
        # Convert Date to datetime
        sales_data['Date'] = pd.to_datetime(sales_data['Date'])
        
        # Merge sales data with store data
        df = pd.merge(sales_data, store_data, on='Store', how='left')
        
        # Handle missing values in store data
        df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
        df['CompetitionOpenSinceMonth'].fillna(df['CompetitionOpenSinceMonth'].median(), inplace=True)
        df['CompetitionOpenSinceYear'].fillna(df['CompetitionOpenSinceYear'].median(), inplace=True)
        df['Promo2SinceWeek'].fillna(df['Promo2SinceWeek'].median(), inplace=True)
        df['Promo2SinceYear'].fillna(df['Promo2SinceYear'].median(), inplace=True)
        
        # Fill categorical missing values
        df['PromoInterval'].fillna('', inplace=True)
        
        # Filter out days when stores were closed (Sales = 0)
        df = df[df['Sales'] > 0]
        
        print(f"Final dataset shape after merging and cleaning: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print("Error: Could not find the Kaggle dataset files (train.csv and store.csv).")
        print("Please download the dataset from https://www.kaggle.com/c/rossmann-store-sales/data")
        print("and place the files in the Data directory.")
        return None

def preprocess_data(data):
    """
    Preprocess the data for forecasting by:
    1. Extracting date features
    2. Creating lag features
    3. Handling categorical variables
    4. Creating interaction features
    5. Scaling numerical features
    
    Args:
        data (DataFrame): Raw data to preprocess
        
    Returns:
        DataFrame: Processed data with engineered features
    """
    df = data.copy()
    
    # Remove any existing one-hot encoded columns to avoid duplicates
    cols_to_drop = [col for col in df.columns if col.startswith('StoreType_') or col.startswith('Assortment_')]
    if cols_to_drop:
        df = df.drop(cols_to_drop, axis=1)
    
    # Step 1: Extract date features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['IsWeekend'] = (df['DayOfWeek'] >= 6).astype(int)  # In Rossmann data, 7=Sunday, 1=Monday
    
    # Step 2: Create lag features
    # Group by store to create store-specific lags
    for store in df['Store'].unique():
        store_data = df[df['Store'] == store].sort_values('Date')
        
        # Previous day sales
        store_data['Sales_Lag1'] = store_data['Sales'].shift(1)
        
        # Previous week sales
        store_data['Sales_Lag7'] = store_data['Sales'].shift(7)
        
        # 7-day moving average
        store_data['Sales_MA7'] = store_data['Sales'].rolling(window=7, min_periods=1).mean()
        
        # Update the main dataframe
        df.loc[df['Store'] == store, 'Sales_Lag1'] = store_data['Sales_Lag1']
        df.loc[df['Store'] == store, 'Sales_Lag7'] = store_data['Sales_Lag7']
        df.loc[df['Store'] == store, 'Sales_MA7'] = store_data['Sales_MA7']
    
    # Handle missing values in lag features
    df['Sales_Lag1'].fillna(df['Sales_Lag1'].mean(), inplace=True)
    df['Sales_Lag7'].fillna(df['Sales_Lag7'].mean(), inplace=True)
    df['Sales_MA7'].fillna(df['Sales_MA7'].mean(), inplace=True)
    
    # Step 3: Handle categorical variables
    # Convert PromoInterval to a more usable feature
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_map = {i+1: months[i] for i in range(12)}
    
    df['MonthStr'] = df['Month'].map(month_map)
    df['IsPromoMonth'] = 0
    
    # Check if current month is in the promotion interval
    for i, row in df.iterrows():
        if row['PromoInterval'] != '' and isinstance(row['PromoInterval'], str):
            if row['MonthStr'] in row['PromoInterval'].split(','):
                df.at[i, 'IsPromoMonth'] = 1
    
    # Drop the original PromoInterval column and temporary MonthStr column
    df = df.drop(['PromoInterval', 'MonthStr'], axis=1)
    
    # One-hot encode remaining categorical variables
    # Use drop_first=False to ensure consistent encoding across different datasets
    df = pd.get_dummies(df, columns=['StoreType', 'Assortment'], drop_first=False)
    
    # Step 4: Create interaction features
    df['Promo_Weekend'] = df['Promo'] * df['IsWeekend']
    df['Promo_SchoolHoliday'] = df['Promo'] * df['SchoolHoliday']
    
    # Step 5: Feature scaling for competition distance
    scaler = StandardScaler()
    df['CompetitionDistance_Scaled'] = scaler.fit_transform(df[['CompetitionDistance']])
    
    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]
    
    return df

def generate_future_dates(last_date, num_days, stores, original_df):
    """
    Generate future dates for forecasting with appropriate features
    
    Args:
        last_date (datetime): Last date in the historical data
        num_days (int): Number of days to forecast
        stores (list): List of store IDs to generate forecasts for
        original_df (DataFrame): Original dataframe with historical data
        
    Returns:
        DataFrame: DataFrame with future dates and features for forecasting
    """
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=num_days)
    future_data = []
    
    for store in stores:
        # Get store information from historical data
        # Use the first row for this store as a template
        store_data = original_df[original_df['Store'] == store]
        if len(store_data) == 0:
            print(f"Warning: No data found for store {store}. Skipping.")
            continue
            
        store_info = store_data.iloc[0].copy()
        
        for date in future_dates:
            # For future dates, we need to make assumptions about promotions, holidays, etc.
            day_of_week = date.weekday() + 1  # Convert to Rossmann format (1=Monday, 7=Sunday)
            if day_of_week == 0:  # Convert Sunday from 0 to 7
                day_of_week = 7
                
            # Create a new row for this date and store
            new_row = {
                'Date': date,
                'Store': store,
                'DayOfWeek': day_of_week,
                'Promo': np.random.choice([0, 1], p=[0.7, 0.3]),  # Assume similar promotion pattern
                'SchoolHoliday': np.random.choice([0, 1], p=[0.8, 0.2])  # Assume similar school holiday pattern
            }
            
            # Add any other columns that exist in the original dataframe
            # This makes the function more robust to different column sets
            for col in original_df.columns:
                if col not in new_row and col not in ['Date', 'Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'Sales', 'Customers']:
                    # For categorical columns that might have been one-hot encoded
                    if col in ['StoreType', 'Assortment']:
                        if col in store_info:
                            new_row[col] = store_info[col]
                    # Handle PromoInterval specially
                    elif col == 'PromoInterval':
                        if col in store_info:
                            # Ensure PromoInterval is a string
                            promo_val = store_info[col]
                            new_row[col] = '' if pd.isna(promo_val) else str(promo_val)
                    # For numerical columns
                    elif col in store_info:
                        new_row[col] = store_info[col]
            
            future_data.append(new_row)
    
    if not future_data:
        raise ValueError("No future data could be generated. Check if store IDs exist in the dataset.")
        
    return pd.DataFrame(future_data)

def split_train_test(df, test_days=14):
    """
    Split the data into training and testing sets based on date
    
    Args:
        df (DataFrame): Processed dataframe
        test_days (int): Number of days to use for testing
        
    Returns:
        tuple: (train_df, test_df) - Training and testing dataframes
    """
    # Use the last test_days days as test set
    split_date = df['Date'].max() - timedelta(days=test_days)
    train_df = df[df['Date'] <= split_date]
    test_df = df[df['Date'] > split_date]
    
    return train_df, test_df

def get_feature_columns(df, exclude_cols=None):
    """
    Get the feature columns for modeling, excluding specified columns
    
    Args:
        df (DataFrame): Processed dataframe
        exclude_cols (list): Columns to exclude from features
        
    Returns:
        list: List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = ['Date', 'Sales', 'DayName', 'Customers']
    
    features = [col for col in df.columns if col not in exclude_cols]
    
    # Verify all features are numeric
    for col in features.copy():
        if df[col].dtype == 'object':
            print(f"Warning: Column {col} is not numeric. Converting to numeric or removing.")
            # Try to convert to numeric, or drop if not possible
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                print(f"Removing column {col} as it cannot be converted to numeric.")
                features.remove(col)
    
    return features