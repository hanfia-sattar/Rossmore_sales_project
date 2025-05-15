import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_forecasted_sales_by_store(forecast_results):
    """
    Plot total forecasted sales by store
    
    Args:
        forecast_results (DataFrame): Forecasting results
    """
    # Aggregate forecast by store
    store_forecast = forecast_results.groupby('Store')['Predicted_Sales'].sum().reset_index()
    store_forecast = store_forecast.sort_values('Predicted_Sales', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(store_forecast['Store'].astype(str), store_forecast['Predicted_Sales'])
    plt.title('Total Forecasted Sales by Store')
    plt.xlabel('Store')
    plt.ylabel('Total Forecasted Sales')
    plt.grid(True, alpha=0.3, axis='y')
    plt.show()
    
    return store_forecast

def plot_forecasted_sales_by_day(forecast_results):
    """
    Plot average forecasted sales by day of week
    
    Args:
        forecast_results (DataFrame): Forecasting results
    """
    # Map day of week to day names
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_mapping = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 
                   5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
    forecast_results['DayName'] = forecast_results['DayOfWeek'].map(day_mapping)
    
    # Calculate average sales by day
    day_forecast = forecast_results.groupby('DayName')['Predicted_Sales'].mean()
    day_forecast = day_forecast.reindex(day_names)
    
    plt.figure(figsize=(10, 6))
    plt.bar(day_forecast.index, day_forecast.values)
    plt.title('Average Forecasted Sales by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Forecasted Sales')
    plt.grid(True, alpha=0.3, axis='y')
    plt.show()
    
    return day_forecast

def plot_historical_vs_forecast(historical_data, forecast_results, store_id):
    """
    Plot historical sales vs forecasted sales for a specific store
    
    Args:
        historical_data (DataFrame): Historical sales data
        forecast_results (DataFrame): Forecasting results
        store_id (int): Store ID to plot
    """
    # Filter data for the specified store
    store_data = historical_data[historical_data['Store'] == store_id]
    store_forecast = forecast_results[forecast_results['Store'] == store_id]
    
    plt.figure(figsize=(14, 7))
    
    # Plot historical sales
    plt.plot(store_data['Date'], store_data['Sales'], 
             label=f'Historical Sales - Store {store_id}', 
             alpha=0.7, linestyle='-')
    
    # Plot forecasted sales
    plt.plot(store_forecast['Date'], store_forecast['Predicted_Sales'], 
             label=f'Forecasted Sales - Store {store_id}', 
             linestyle='--', marker='o', color='red')
    
    plt.title(f'Sales Forecast for Store {store_id}')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_promotion_impact(forecast_results):
    """
    Analyze the impact of promotions on sales
    
    Args:
        forecast_results (DataFrame): Forecasting results
        
    Returns:
        float: Percentage lift in sales due to promotions
    """
    # Group by promotion status and calculate average sales
    promo_impact = forecast_results.groupby('Promo')['Predicted_Sales'].mean()
    
    if len(promo_impact) > 1:
        promo_lift = ((promo_impact[1] / promo_impact[0]) - 1) * 100
        
        # Plot promotion impact
        plt.figure(figsize=(8, 6))
        plt.bar(['No Promotion', 'Promotion'], [promo_impact[0], promo_impact[1]])
        plt.title('Average Sales by Promotion Status')
        plt.ylabel('Average Sales')
        plt.grid(True, alpha=0.3, axis='y')
        plt.show()
        
        return promo_lift
    else:
        print("Not enough data to analyze promotion impact.")
        return None

def analyze_weekend_vs_weekday(forecast_results):
    """
    Analyze weekend vs weekday sales
    
    Args:
        forecast_results (DataFrame): Forecasting results
        
    Returns:
        float: Percentage difference between weekend and weekday sales
    """
    # Map day of week to day names
    day_mapping = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 
                   5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
    forecast_results['DayName'] = forecast_results['DayOfWeek'].map(day_mapping)
    
    # Define weekend days
    weekend_days = ['Saturday', 'Sunday']
    
    # Calculate average sales for weekends and weekdays
    weekend_sales = forecast_results[forecast_results['DayName'].isin(weekend_days)]['Predicted_Sales'].mean()
    weekday_sales = forecast_results[~forecast_results['DayName'].isin(weekend_days)]['Predicted_Sales'].mean()
    
    # Calculate percentage difference
    weekend_lift = ((weekend_sales / weekday_sales) - 1) * 100
    
    # Plot weekend vs weekday sales
    plt.figure(figsize=(8, 6))
    plt.bar(['Weekday', 'Weekend'], [weekday_sales, weekend_sales])
    plt.title('Average Sales: Weekday vs Weekend')
    plt.ylabel('Average Sales')
    plt.grid(True, alpha=0.3, axis='y')
    plt.show()
    
    return weekend_lift

def generate_business_insights(historical_data, forecast_results, store_id=None):
    """
    Generate comprehensive business insights from the forecast
    
    Args:
        historical_data (DataFrame): Historical sales data
        forecast_results (DataFrame): Forecasting results
        store_id (int): Store ID to highlight, if None, uses the top performing store
    """
    print("\nGenerating business insights from sales forecast...")
    
    # Plot forecasted sales by store
    store_forecast = plot_forecasted_sales_by_store(forecast_results)
    
    # Plot forecasted sales by day
    day_forecast = plot_forecasted_sales_by_day(forecast_results)
    
    # If store_id is not provided, use the top performing store
    if store_id is None:
        store_id = store_forecast.iloc[0]['Store']
    
    # Plot historical vs forecast for the selected store
    plot_historical_vs_forecast(historical_data, forecast_results, store_id)
    
    # Analyze promotion impact
    promo_lift = analyze_promotion_impact(forecast_results)
    
    # Analyze weekend vs weekday
    weekend_lift = analyze_weekend_vs_weekday(forecast_results)
    
    # Generate business insights
    print("\nBusiness Insights from Sales Forecast:")
    print("--------------------------------------")
    
    # 1. Top performing stores
    top_store = store_forecast.iloc[0]['Store']
    print(f"1. Store {top_store} is forecasted to have the highest sales.")
    
    # 2. Best days for sales
    best_day = day_forecast.idxmax()
    print(f"2. {best_day} is forecasted to be the best day for sales across all stores.")
    
    # 3. Promotion impact
    if promo_lift is not None:
        print(f"3. Promotions are forecasted to increase sales by {promo_lift:.2f}%.")
    
    # 4. Weekend vs. Weekday
    print(f"4. Weekend sales are forecasted to be {weekend_lift:.2f}% {'higher' if weekend_lift > 0 else 'lower'} than weekday sales.")
    
    # Generate recommendations
    print("\nRecommendations:")
    print(f"1. Optimize inventory levels for Store {top_store} to meet the high forecasted demand.")
    print(f"2. Schedule more staff on {best_day} to handle increased customer traffic.")
    
    if promo_lift is not None and promo_lift > 0:
        print("3. Consider running more promotions as they significantly boost sales.")
    
    print(f"4. Prepare for {'higher' if weekend_lift > 0 else 'lower'} weekend sales with appropriate staffing and inventory.")
    
    return {
        'top_store': top_store,
        'best_day': best_day,
        'promo_lift': promo_lift,
        'weekend_lift': weekend_lift
    }

