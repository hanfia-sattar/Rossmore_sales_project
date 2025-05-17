import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import time
import os
from datetime import datetime, timedelta

# Import custom modules
from data_preprocessing import load_rossmann_data, preprocess_data, get_feature_columns, split_train_test, generate_future_dates
from exploratory_analysis import plot_sales_by_store, plot_sales_by_day_of_week, plot_sales_by_store_type, plot_sales_by_month, plot_sales_vs_promo, plot_correlation_matrix
from forecasting_pipeline import train_models, evaluate_all_models, forecast_sales, save_model_and_results
from business_insights import analyze_promotion_impact, analyze_weekend_vs_weekday, plot_forecasted_sales_by_store, plot_forecasted_sales_by_day

# Set page configuration
st.set_page_config(
    page_title="Rossmann Sales Forecasting Dashboard",
    page_icon="Data\statistic-icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E3F2FD;
    }
    .subsection-header {
        font-size: 1.4rem;
        font-weight: 500;
        color: #1565C0;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .metric-card {
        background-color: #E3F2FD;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-card {
        background-color: #F5F5F5;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation and controls
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center;">
        <img src="https://png.pngtree.com/png-vector/20220701/ourmid/pngtree-statistic-icon-vector-logo-template-illustration-design-png-image_5676536.png" width="50" style="margin-right: 5px;">
        <h1 style="margin: 0;">Rossmann Insights</h1>
    </div>
    """,
    unsafe_allow_html=True
) 
st.sidebar.markdown("## Navigation")

# Create navigation options
pages = ["Overview", "Data Exploration", "Data Preprocessing", "Model Training", "Forecasting", "Business Insights"]
selected_page = st.sidebar.radio("Go to", pages)

# Sidebar filters (will be used across pages)
st.sidebar.markdown("## Filters")

# Initialize session state for storing data between pages
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.df_subset = None
    st.session_state.processed_df = None
    st.session_state.features = None
    st.session_state.models = None
    st.session_state.best_model = None
    st.session_state.forecast_results = None
    st.session_state.store_ids = None
    st.session_state.insights = None

# Function to load data
@st.cache_data
def load_data():
    df = load_rossmann_data()
    return df

# Function to create plotly figures from matplotlib figures
def plt_to_plotly():
    fig = plt.gcf()
    plotly_fig = go.Figure()
    for ax in fig.axes:
        for line in ax.lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            plotly_fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=line.get_label()))
    plt.close(fig)
    return plotly_fig

# Overview Page
if selected_page == "Overview":
    st.markdown("<div class='main-header' style='font-size:32px; font-weight:bold;'>Sales Forecasting & Recommendation Dashboard</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This dashboard demonstrates a complete sales forecasting pipeline using the Kaggle Rossmann Store Sales dataset.
        
        The Rossmann dataset contains historical sales data for 1,115 Rossmann stores. The task is to forecast the sales
        for these stores in the future.
        
        ### Process Overview:
        1. **Data Loading**: Load and prepare the Kaggle dataset
        2. **Exploratory Analysis**: Analyze sales patterns and relationships
        3. **Data Preprocessing**: Clean data and engineer features
        4. **Model Training**: Train and evaluate forecasting models
        5. **Forecasting**: Generate sales forecasts
        6. **Business Insights**: Extract actionable insights
        """)
    
    with col2:
        st.image("Data\statistic-icon.png", width=300)
    
    # Load data button
    if st.button("Load Dataset"):
        with st.spinner("Loading Rossmann Store Sales data..."):
            df = load_data()
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                
                # Select a subset of stores for demonstration
                store_ids = df['Store'].unique()[:5]  # First 5 stores
                st.session_state.store_ids = store_ids
                st.session_state.df_subset = df[df['Store'].isin(store_ids)]
                
                st.success(f"Data loaded successfully! Dataset contains {len(df)} rows and {df.shape[1]} columns.")
                st.info("A subset of 5 stores has been selected for demonstration purposes.")
            else:
                st.error("Error loading data. Please check if the dataset files exist in the Data directory.")
    
    # Display dataset preview if loaded
    if st.session_state.data_loaded:
        st.markdown("<div class='section-header'>Dataset Preview</div>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Sales Data", "Store Data"])
        
        with tab1:
            sales_cols = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
            st.dataframe(st.session_state.df_subset[sales_cols].head(10))
        
        with tab2:
            store_cols = ['Store', 'StoreType', 'Assortment', 'CompetitionDistance', 'Promo2']
            st.dataframe(st.session_state.df_subset[store_cols].drop_duplicates().head(10))
        
        # Display basic statistics
        st.markdown("<div class='subsection-header'>Basic Statistics</div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stores", f"{st.session_state.df['Store'].nunique():,}")
        
        with col2:
            st.metric("Date Range", f"{st.session_state.df['Date'].min().strftime('%Y-%m-%d')} to {st.session_state.df['Date'].max().strftime('%Y-%m-%d')}")
        
        with col3:
            st.metric("Average Daily Sales", f"${st.session_state.df['Sales'].mean():,.2f}")
        
        with col4:
            st.metric("Total Sales", f"${st.session_state.df['Sales'].sum():,.2f}")

# Data Exploration Page
elif selected_page == "Data Exploration":
    st.markdown("<div class='main-header'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load the dataset first from the Overview page.")
    else:
        st.markdown("Explore the patterns and relationships in the Rossmann sales data.")
        
        # Store selection for exploration
        available_stores = st.session_state.store_ids
        selected_stores = st.sidebar.multiselect(
            "Select Stores for Analysis",
            options=available_stores,
            default=available_stores[:3]
        )
        
        if not selected_stores:
            st.warning("Please select at least one store for analysis.")
        else:
            # Filter data for selected stores
            df_selected = st.session_state.df_subset[st.session_state.df_subset['Store'].isin(selected_stores)]
            
            # Sales Distribution by Store
            st.markdown("<div class='section-header'>Sales Distribution by Store</div>", unsafe_allow_html=True)
            
            fig = px.box(df_selected, x='Store', y='Sales', title='Sales Distribution by Store')
            fig.update_layout(xaxis_title='Store', yaxis_title='Sales')
            st.plotly_chart(fig, use_container_width=True)
            
            # Sales by Day of Week
            st.markdown("<div class='section-header'>Sales by Day of Week</div>", unsafe_allow_html=True)
            
            # Map day of week to day names
            day_mapping = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 
                          5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
            df_selected['DayName'] = df_selected['DayOfWeek'].map(day_mapping)
            
            fig = px.box(df_selected, x='DayName', y='Sales', 
                        category_orders={"DayName": ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']},
                        title='Sales Distribution by Day of Week')
            fig.update_layout(xaxis_title='Day of Week', yaxis_title='Sales')
            st.plotly_chart(fig, use_container_width=True)
            
            # Sales by Store Type
            st.markdown("<div class='section-header'>Sales by Store Type</div>", unsafe_allow_html=True)
            
            fig = px.box(df_selected, x='StoreType', y='Sales', title='Sales Distribution by Store Type')
            fig.update_layout(xaxis_title='Store Type', yaxis_title='Sales')
            st.plotly_chart(fig, use_container_width=True)
            
            # Sales Trend Over Time
            st.markdown("<div class='section-header'>Sales Trend Over Time</div>", unsafe_allow_html=True)
            
            # Aggregate sales by date
            daily_sales = df_selected.groupby('Date')['Sales'].mean().reset_index()
            
            fig = px.line(daily_sales, x='Date', y='Sales', title='Average Daily Sales Over Time')
            fig.update_layout(xaxis_title='Date', yaxis_title='Average Sales')
            st.plotly_chart(fig, use_container_width=True)
            
            # Sales vs Promotion
            st.markdown("<div class='section-header'>Impact of Promotions on Sales</div>", unsafe_allow_html=True)
            
            fig = px.box(df_selected, x='Promo', y='Sales', title='Sales Distribution by Promotion Status')
            fig.update_layout(xaxis_title='Promotion (0=No, 1=Yes)', yaxis_title='Sales')
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation Matrix
            st.markdown("<div class='section-header'>Correlation Matrix</div>", unsafe_allow_html=True)
            
            # Select only numeric columns
            numeric_df = df_selected.select_dtypes(include=[np.number])
            
            # Calculate correlation matrix
            corr = numeric_df.corr()
            
            fig = px.imshow(corr, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r')
            fig.update_layout(title='Correlation Matrix of Numerical Features')
            st.plotly_chart(fig, use_container_width=True)
            
            # Key Findings
            st.markdown("<div class='section-header'>Key Findings from EDA</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
                st.markdown("**Sales Variation by Store Type**")
                st.markdown("Store types show significant variation in sales performance, with some types consistently outperforming others.")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
                st.markdown("**Day of Week Patterns**")
                st.markdown("Sales show clear patterns across days of the week, with weekends typically showing different patterns than weekdays.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
                st.markdown("**Promotion Impact**")
                st.markdown("Promotions generally have a positive impact on sales, with promoted days showing higher average sales.")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
                st.markdown("**Customer and Sales Correlation**")
                st.markdown("There's a strong positive correlation between the number of customers and sales, indicating that increasing foot traffic is key to sales growth.")
                st.markdown("</div>", unsafe_allow_html=True)

# Data Preprocessing Page
elif selected_page == "Data Preprocessing":
    st.markdown("<div class='main-header'>Data Preprocessing</div>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load the dataset first from the Overview page.")
    else:
        st.markdown("""
        This step involves cleaning the data and engineering features to prepare for modeling.
        
        The preprocessing pipeline includes:
        - Handling missing values
        - Extracting date features
        - Creating lag features
        - Handling categorical variables
        - Creating interaction features
        """)
        
        if st.button("Run Data Preprocessing"):
            with st.spinner("Preprocessing data..."):
                # Process the data
                processed_df = preprocess_data(st.session_state.df_subset)
                st.session_state.processed_df = processed_df
                
                # Get feature columns for modeling
                features = get_feature_columns(processed_df)
                st.session_state.features = features
                
                st.success("Data preprocessing completed successfully!")
        
        if st.session_state.processed_df is not None:
            # Display the processed data
            st.markdown("<div class='section-header'>Processed Data Preview</div>", unsafe_allow_html=True)
            st.dataframe(st.session_state.processed_df.head())
            
            # Show feature engineering results
            st.markdown("<div class='section-header'>Feature Engineering Results</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='subsection-header'>Original Features</div>", unsafe_allow_html=True)
                original_features = st.session_state.df_subset.columns.tolist()
                st.write(f"Number of original features: {len(original_features)}")
                st.write(original_features)
            
            with col2:
                st.markdown("<div class='subsection-header'>Engineered Features</div>", unsafe_allow_html=True)
                new_features = [col for col in st.session_state.processed_df.columns if col not in st.session_state.df_subset.columns]
                st.write(f"Number of new features: {len(new_features)}")
                st.write(new_features)
            
            # Display the features that will be used for modeling
            st.markdown("<div class='section-header'>Features for Modeling</div>", unsafe_allow_html=True)
            st.write(f"Number of features for modeling: {len(st.session_state.features)}")
            
            # Display features in a more readable format with columns
            feature_cols = st.columns(3)
            features_per_col = len(st.session_state.features) // 3 + 1
            
            for i, col in enumerate(feature_cols):
                start_idx = i * features_per_col
                end_idx = min((i + 1) * features_per_col, len(st.session_state.features))
                col.write(st.session_state.features[start_idx:end_idx])
            
            # Show data split information
            st.markdown("<div class='section-header'>Train-Test Split</div>", unsafe_allow_html=True)
            
            train_df, test_df = split_train_test(st.session_state.processed_df, test_days=14)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Training Set Size", f"{len(train_df):,} samples")
                st.write(f"Date range: {train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')}")
            
            with col2:
                st.metric("Testing Set Size", f"{len(test_df):,} samples")
                st.write(f"Date range: {test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')}")

# Model Training Page
elif selected_page == "Model Training":
    st.markdown("<div class='main-header'>Model Training</div>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load the dataset first from the Overview page.")
    elif st.session_state.processed_df is None:
        st.warning("Please run data preprocessing first from the Data Preprocessing page.")
    else:
        st.markdown("""
        This step involves training multiple machine learning models to forecast sales.
        
        We'll train and evaluate the following models:
        - Linear Regression
        - Random Forest
        - XGBoost
        """)
        
        if st.button("Train Models"):
            with st.spinner("Training models..."):
                # Split data into training and testing sets
                train_df, test_df = split_train_test(st.session_state.processed_df, test_days=14)
                
                # Prepare features and target
                X_train = train_df[st.session_state.features]
                y_train = train_df['Sales']
                X_test = test_df[st.session_state.features]
                y_test = test_df['Sales']
                
                # Train models
                models = train_models(X_train, y_train)
                st.session_state.models = models
                
                # Evaluate models
                results, best_model_name = evaluate_all_models(models, X_test, y_test)
                st.session_state.best_model = models[best_model_name]
                st.session_state.model_results = results
                st.session_state.best_model_name = best_model_name
                
                st.success(f"Models trained successfully! Best model: {best_model_name}")
        
        if st.session_state.models is not None:
            # Display model evaluation results
            st.markdown("<div class='section-header'>Model Evaluation Results</div>", unsafe_allow_html=True)
            
            # Create a dataframe for model comparison
            model_comparison = []
            for model_name, result in st.session_state.model_results.items():
                model_comparison.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'RMSE': result['rmse'],
                    'MAE': result['mae'],
                    'MAPE (%)': result['mape'],
                    'R² Score': result['r2']
                })
            
            model_df = pd.DataFrame(model_comparison)
            
            # Highlight the best model
            best_model_name_display = st.session_state.best_model_name.replace('_', ' ').title()
            
            # Display model comparison table
            st.dataframe(model_df.style.highlight_min(subset=['RMSE', 'MAE', 'MAPE (%)'], color='#E3F2FD')
                                    .highlight_max(subset=['R² Score'], color='#E3F2FD'))
            
            # Create visualizations for model comparison
            st.markdown("<div class='section-header'>Model Performance Comparison</div>", unsafe_allow_html=True)
            
            # RMSE Comparison
            fig = px.bar(model_df, x='Model', y='RMSE', title='Root Mean Squared Error (RMSE) by Model',
                        color='Model', text_auto='.2f')
            fig.update_layout(xaxis_title='Model', yaxis_title='RMSE (lower is better)')
            st.plotly_chart(fig, use_container_width=True)
            
            # R² Score Comparison
            fig = px.bar(model_df, x='Model', y='R² Score', title='R² Score by Model',
                        color='Model', text_auto='.4f')
            fig.update_layout(xaxis_title='Model', yaxis_title='R² Score (higher is better)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance (for Random Forest and XGBoost)
            st.markdown("<div class='section-header'>Feature Importance</div>", unsafe_allow_html=True)
            
            if 'random_forest' in st.session_state.models:
                rf_model = st.session_state.models['random_forest']
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state.features,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(feature_importance.head(15), x='Importance', y='Feature', 
                            title='Top 15 Features (Random Forest)',
                            orientation='h')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Best Model Details
            st.markdown("<div class='section-header'>Best Model Details</div>", unsafe_allow_html=True)
            
            st.markdown(f"<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(f"### {best_model_name_display}")
            
            best_result = st.session_state.model_results[st.session_state.best_model_name]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("RMSE", f"{best_result['rmse']:.2f}")
            
            with col2:
                st.metric("MAE", f"{best_result['mae']:.2f}")
            
            with col3:
                st.metric("MAPE", f"{best_result['mape']:.2f}%")
            
            with col4:
                st.metric("R² Score", f"{best_result['r2']:.4f}")
            
            st.markdown("</div>", unsafe_allow_html=True)

# Forecasting Page
elif selected_page == "Forecasting":
    st.markdown("<div class='main-header'>Sales Forecasting</div>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load the dataset first from the Overview page.")
    elif st.session_state.processed_df is None:
        st.warning("Please run data preprocessing first from the Data Preprocessing page.")
    elif st.session_state.best_model is None:
        st.warning("Please train models first from the Model Training page.")
    else:
        st.markdown("""
        This step generates sales forecasts for future dates using the best trained model.
        """)
        
        # Forecast settings
        st.markdown("<div class='section-header'>Forecast Settings</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_days = st.slider("Number of Days to Forecast", min_value=7, max_value=30, value=14, step=1)
        
        with col2:
            available_stores = st.session_state.store_ids
            forecast_stores = st.multiselect(
                "Select Stores for Forecasting",
                options=available_stores,
                default=available_stores[:3]
            )
        
        if not forecast_stores:
            st.warning("Please select at least one store for forecasting.")
        else:
            if st.button("Generate Forecast"):
                with st.spinner(f"Generating {forecast_days}-day forecast for {len(forecast_stores)} stores..."):
                    # Generate future dates for forecasting
                    last_date = st.session_state.processed_df['Date'].max()
                    
                    # Use the original dataframe for future date generation
                    future_df = generate_future_dates(last_date, forecast_days, forecast_stores, st.session_state.df_subset)
                    
                    # Forecast sales using the best model
                    forecast_results = forecast_sales(st.session_state.best_model, 
                                                     st.session_state.processed_df, 
                                                     future_df, 
                                                     st.session_state.features)
                    
                    st.session_state.forecast_results = forecast_results
                    
                    st.success(f"Sales forecast generated successfully for {len(forecast_stores)} stores over {forecast_days} days!")
            
            if st.session_state.forecast_results is not None:
                # Display forecast results
                st.markdown("<div class='section-header'>Forecast Results</div>", unsafe_allow_html=True)
                
                # Filter forecast results for selected stores
                filtered_forecast = st.session_state.forecast_results[st.session_state.forecast_results['Store'].isin(forecast_stores)]
                
                # Display forecast table
                st.dataframe(filtered_forecast[['Store', 'Date', 'DayOfWeek', 'Predicted_Sales']].sort_values(['Store', 'Date']))
                
                # Visualize forecast
                st.markdown("<div class='section-header'>Forecast Visualization</div>", unsafe_allow_html=True)
                
                # Total forecasted sales by store
                store_forecast = filtered_forecast.groupby('Store')['Predicted_Sales'].sum().reset_index()
                
                fig = px.bar(store_forecast, x='Store', y='Predicted_Sales', 
                            title='Total Forecasted Sales by Store',
                            color='Store')
                fig.update_layout(xaxis_title='Store', yaxis_title='Total Forecasted Sales')
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecasted sales by day
                # Map day of week to day names
                day_mapping = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 
                              5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
                filtered_forecast['DayName'] = filtered_forecast['DayOfWeek'].map(day_mapping)
                
                # Calculate average sales by day
                day_forecast = filtered_forecast.groupby('DayName')['Predicted_Sales'].mean().reset_index()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_forecast['DayName'] = pd.Categorical(day_forecast['DayName'], categories=day_order, ordered=True)
                day_forecast = day_forecast.sort_values('DayName')
                
                fig = px.bar(day_forecast, x='DayName', y='Predicted_Sales', 
                            title='Average Forecasted Sales by Day of Week',
                            color='DayName')
                fig.update_layout(xaxis_title='Day of Week', yaxis_title='Average Forecasted Sales')
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast timeline for each store
                st.markdown("<div class='section-header'>Forecast Timeline by Store</div>", unsafe_allow_html=True)
                
                selected_store = st.selectbox("Select Store to View Forecast Timeline", options=forecast_stores)
                
                store_timeline = filtered_forecast[filtered_forecast['Store'] == selected_store]
                
                fig = px.line(store_timeline, x='Date', y='Predicted_Sales', 
                             title=f'Sales Forecast Timeline for Store {selected_store}',
                             markers=True)
                fig.update_layout(xaxis_title='Date', yaxis_title='Forecasted Sales')
                st.plotly_chart(fig, use_container_width=True)
                
                # Compare with historical data if available
                if st.checkbox("Compare with Historical Data"):
                    # Get historical data for the selected store
                    historical_data = st.session_state.df_subset[st.session_state.df_subset['Store'] == selected_store]
                    
                    # Create a combined plot
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=historical_data['Date'],
                        y=historical_data['Sales'],
                        mode='lines',
                        name='Historical Sales',
                        line=dict(color='blue')
                    ))
                    
                    # Add forecast data
                    fig.add_trace(go.Scatter(
                        x=store_timeline['Date'],
                        y=store_timeline['Predicted_Sales'],
                        mode='lines+markers',
                        name='Forecasted Sales',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f'Historical vs Forecasted Sales for Store {selected_store}',
                        xaxis_title='Date',
                        yaxis_title='Sales',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

# Business Insights Page
elif selected_page == "Business Insights":
    st.markdown("<div class='main-header'>Business Insights</div>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load the dataset first from the Overview page.")
    elif st.session_state.processed_df is None:
        st.warning("Please run data preprocessing first from the Data Preprocessing page.")
    elif st.session_state.best_model is None:
        st.warning("Please train models first from the Model Training page.")
    elif st.session_state.forecast_results is None:
        st.warning("Please generate forecasts first from the Forecasting page.")
    else:
        st.markdown("""
        This step extracts actionable business insights from the sales forecasts.
        """)
        
        if st.button("Generate Business Insights"):
            with st.spinner("Generating business insights..."):
                # Analyze promotion impact
                promo_lift = analyze_promotion_impact(st.session_state.forecast_results)
                
                # Analyze weekend vs weekday
                weekend_lift = analyze_weekend_vs_weekday(st.session_state.forecast_results)
                
                # Store forecast by store
                store_forecast = plot_forecasted_sales_by_store(st.session_state.forecast_results)
                
                # Day forecast
                day_forecast = plot_forecasted_sales_by_day(st.session_state.forecast_results)
                
                # Find top performing store
                top_store = store_forecast.iloc[0]['Store']
                
                # Find best day for sales
                best_day = day_forecast.idxmax()
                
                # Store insights in session state
                st.session_state.insights = {
                    'top_store': top_store,
                    'best_day': best_day,
                    'promo_lift': promo_lift,
                    'weekend_lift': weekend_lift,
                    'store_forecast': store_forecast,
                    'day_forecast': day_forecast
                }
                
                st.success("Business insights generated successfully!")
        
        if st.session_state.insights is not None:
            # Display business insights
            st.markdown("<div class='section-header'>Key Business Insights</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"### Top Performing Store")
                st.metric("Store ID", f"{st.session_state.insights['top_store']}")
                st.markdown(f"Store {st.session_state.insights['top_store']} is forecasted to have the highest sales.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"### Best Day for Sales")
                st.metric("Day", f"{st.session_state.insights['best_day']}")
                st.markdown(f"{st.session_state.insights['best_day']} is forecasted to be the best day for sales across all stores.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"### Promotion Impact")
                if st.session_state.insights['promo_lift'] is not None:
                    st.metric("Sales Lift", f"{st.session_state.insights['promo_lift']:.2f}%")
                    st.markdown(f"Promotions are forecasted to increase sales by {st.session_state.insights['promo_lift']:.2f}%.")
                else:
                    st.markdown("Not enough data to analyze promotion impact.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"### Weekend vs Weekday")
                st.metric("Weekend Lift", f"{st.session_state.insights['weekend_lift']:.2f}%")
                st.markdown(f"Weekend sales are forecasted to be {st.session_state.insights['weekend_lift']:.2f}% {'higher' if st.session_state.insights['weekend_lift'] > 0 else 'lower'} than weekday sales.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Visualizations
            st.markdown("<div class='section-header'>Insight Visualizations</div>", unsafe_allow_html=True)
            
            # Total forecasted sales by store
            fig = px.bar(st.session_state.insights['store_forecast'], x='Store', y='Predicted_Sales',
                        title='Total Forecasted Sales by Store',
                        color='Predicted_Sales', color_continuous_scale='Viridis')
            fig.update_layout(xaxis_title='Store', yaxis_title='Total Forecasted Sales')
            st.plotly_chart(fig, use_container_width=True)
            
            # Average forecasted sales by day
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_values = [st.session_state.insights['day_forecast'].get(day, 0) for day in day_names]
            
            fig = px.bar(x=day_names, y=day_values,
                        title='Average Forecasted Sales by Day of Week',
                        color=day_values, color_continuous_scale='Viridis')
            fig.update_layout(xaxis_title='Day of Week', yaxis_title='Average Forecasted Sales')
            st.plotly_chart(fig, use_container_width=True)
            
            # Promotion impact
            if st.session_state.insights['promo_lift'] is not None:
                # Group by promotion status and calculate average sales
                promo_impact = st.session_state.forecast_results.groupby('Promo')['Predicted_Sales'].mean().reset_index()
                
                fig = px.bar(promo_impact, x='Promo', y='Predicted_Sales',
                            title='Average Sales by Promotion Status',
                            color='Promo', color_discrete_map={0: '#1E88E5', 1: '#4CAF50'})
                fig.update_layout(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=[0, 1],
                        ticktext=['No Promotion', 'Promotion']
                    ),
                    xaxis_title='Promotion Status',
                    yaxis_title='Average Sales'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Weekend vs weekday
            # Calculate average sales for weekends and weekdays
            day_mapping = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 
                          5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
            st.session_state.forecast_results['DayName'] = st.session_state.forecast_results['DayOfWeek'].map(day_mapping)
            
            weekend_days = ['Saturday', 'Sunday']
            weekend_sales = st.session_state.forecast_results[st.session_state.forecast_results['DayName'].isin(weekend_days)]['Predicted_Sales'].mean()
            weekday_sales = st.session_state.forecast_results[~st.session_state.forecast_results['DayName'].isin(weekend_days)]['Predicted_Sales'].mean()
            
            weekend_data = pd.DataFrame({
                'Day Type': ['Weekday', 'Weekend'],
                'Average Sales': [weekday_sales, weekend_sales]
            })
            
            fig = px.bar(weekend_data, x='Day Type', y='Average Sales',
                        title='Average Sales: Weekday vs Weekend',
                        color='Day Type', color_discrete_map={'Weekday': '#1E88E5', 'Weekend': '#4CAF50'})
            fig.update_layout(xaxis_title='Day Type', yaxis_title='Average Sales')
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("<div class='section-header'>Business Recommendations</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
            st.markdown(f"**1. Optimize inventory levels for Store {st.session_state.insights['top_store']}**")
            st.markdown(f"Store {st.session_state.insights['top_store']} is forecasted to have the highest sales. Ensure adequate inventory levels to meet the high forecasted demand.")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
            st.markdown(f"**2. Schedule more staff on {st.session_state.insights['best_day']}**")
            st.markdown(f"{st.session_state.insights['best_day']} is forecasted to be the best day for sales. Schedule more staff to handle increased customer traffic.")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if st.session_state.insights['promo_lift'] is not None and st.session_state.insights['promo_lift'] > 0:
                st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
                st.markdown("**3. Run more promotions**")
                st.markdown(f"Promotions are forecasted to increase sales by {st.session_state.insights['promo_lift']:.2f}%. Consider running more promotions to boost sales.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
            st.markdown("**4. Weekend staffing and inventory**")
            st.markdown(f"Prepare for {'higher' if st.session_state.insights['weekend_lift'] > 0 else 'lower'} weekend sales with appropriate staffing and inventory levels.")
            st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Rossmann Sales Forecasting Dashboard | Developed by Hanfia")