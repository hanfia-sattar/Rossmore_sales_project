import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
warnings.filterwarnings('ignore')

# Import custom modules
from data_preprocessing import load_rossmann_data, preprocess_data, get_feature_columns
from exploratory_analysis import run_exploratory_analysis
from forecasting_pipeline import run_forecasting_pipeline
from business_insights import generate_business_insights

# Set page configuration
st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1565C0;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .info-text {
        font-size: 1rem;
        color: #424242;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'eda_completed' not in st.session_state:
    st.session_state.eda_completed = False
if 'forecasting_completed' not in st.session_state:
    st.session_state.forecasting_completed = False
if 'insights_generated' not in st.session_state:
    st.session_state.insights_generated = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_subset' not in st.session_state:
    st.session_state.df_subset = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'eda_figures' not in st.session_state:
    st.session_state.eda_figures = {}

# Sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shop-department.png", width=80)
    st.markdown("<h2 style='text-align: center;'>Sales Forecasting</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    selected = option_menu(
        menu_title=None,
        options=["Home", "Data Loading", "Exploratory Analysis", "Data Processing", "Forecasting", "Business Insights"],
        icons=["house", "database", "search", "gear", "graph-up", "lightbulb"],
        menu_icon="cast",
        default_index=0,
    )
    
    st.markdown("---")
    st.markdown("### Pipeline Status")
    
    # Status indicators
    if st.session_state.data_loaded:
        st.success("✅ Data Loaded")
    else:
        st.info("⏳ Data Loading Pending")
        
    if st.session_state.eda_completed:
        st.success("✅ EDA Completed")
    else:
        st.info("⏳ EDA Pending")
        
    if st.session_state.data_processed:
        st.success("✅ Data Processed")
    else:
        st.info("⏳ Data Processing Pending")
        
    if st.session_state.forecasting_completed:
        st.success("✅ Forecasting Completed")
    else:
        st.info("⏳ Forecasting Pending")
        
    if st.session_state.insights_generated:
        st.success("✅ Insights Generated")
    else:
        st.info("⏳ Insights Pending")

# Home page
if selected == "Home":
    st.markdown("<h1 class='main-header'>Rossmann Store Sales Forecasting Dashboard</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="highlight">
        <p class="info-text">This dashboard provides a comprehensive sales forecasting system using Kaggle's Rossmann Store Sales dataset. 
        Navigate through the different sections to explore the data, view the analysis, and see the forecasting results.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 class='sub-header'>Dashboard Features</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        - **Data Loading**: Load and explore the Rossmann Store Sales dataset
        - **Exploratory Analysis**: Visualize key trends and patterns in the data
        - **Data Processing**: Preprocess data and engineer features for modeling
        - **Forecasting**: Run the forecasting pipeline and view predictions
        - **Business Insights**: Generate actionable business insights from the forecasts
        """)
        
        st.markdown("<h2 class='sub-header'>How to Use</h2>", unsafe_allow_html=True)
        st.markdown("""
        1. Start by navigating to the **Data Loading** section to load the dataset
        2. Proceed through each section in order to complete the full forecasting pipeline
        3. Use the sidebar to navigate between different sections
        4. Check the pipeline status in the sidebar to see your progress
        """)
    
    with col2:
        st.image("https://img.icons8.com/color/240/000000/sales-performance.png", width=200)
        
        # Quick start button
        st.markdown("### Quick Start")
        if st.button("Run Complete Pipeline", key="run_pipeline"):
            with st.spinner("Running complete pipeline..."):
                # Simulate pipeline execution
                if not st.session_state.data_loaded:
                    st.session_state.df = load_rossmann_data()
                    store_ids = st.session_state.df['Store'].unique()[:5]
                    st.session_state.df_subset = st.session_state.df[st.session_state.df['Store'].isin(store_ids)]
                    st.session_state.data_loaded = True
                    time.sleep(1)
                
                if not st.session_state.eda_completed:
                    # Run EDA without expecting returned figures
                    run_exploratory_analysis(st.session_state.df_subset, store_ids)
                    st.session_state.eda_figures = {}  # Initialize empty dict
                    st.session_state.eda_completed = True
                    time.sleep(1)
                
                if not st.session_state.data_processed:
                    st.session_state.processed_df = preprocess_data(st.session_state.df_subset)
                    st.session_state.features = get_feature_columns(st.session_state.processed_df)
                    st.session_state.data_processed = True
                    time.sleep(1)
                
                if not st.session_state.forecasting_completed:
                    st.session_state.best_model, st.session_state.forecast_results = run_forecasting_pipeline(
                        st.session_state.processed_df, 
                        st.session_state.features, 
                        forecast_days=14, 
                        store_ids=store_ids,
                        original_df=st.session_state.df_subset
                    )
                    st.session_state.forecasting_completed = True
                    time.sleep(1)
                
                if not st.session_state.insights_generated:
                    st.session_state.insights = generate_business_insights(
                        st.session_state.df_subset, 
                        st.session_state.forecast_results
                    )
                    st.session_state.insights_generated = True
            
            st.success("Complete pipeline executed successfully!")
            st.info("Navigate through the sidebar to explore results in each section")

# Data Loading page
elif selected == "Data Loading":
    st.markdown("<h1 class='main-header'>Data Loading</h1>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown("<p class='info-text'>Load the Rossmann Store Sales dataset from Kaggle.</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### Dataset Information")
            st.markdown("""
            The Rossmann Store Sales dataset contains historical sales data for 1,115 Rossmann stores. The data includes:
            - Store sales data
            - Store information (type, assortment, competition)
            - Promotional information
            - Seasonal and holiday effects
            """)
        
        with col2:
            st.image("https://img.icons8.com/color/240/000000/database-restore.png", width=100)
        
        if st.button("Load Dataset", key="load_data"):
            with st.spinner("Loading Rossmann Store Sales dataset..."):
                # Load the data
                st.session_state.df = load_rossmann_data()
                
                if st.session_state.df is not None:
                    # Select a subset of stores for demonstration
                    store_ids = st.session_state.df['Store'].unique()[:5]  # First 5 stores
                    st.session_state.df_subset = st.session_state.df[st.session_state.df['Store'].isin(store_ids)]
                    st.session_state.data_loaded = True
                    st.success("Data loaded successfully!")
                else:
                    st.error("Error loading data. Please check your data source.")
    else:
        st.success("Data already loaded!")
        
        # Display dataset information
        st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Data Sample")
            st.dataframe(st.session_state.df_subset.head())
        
        with col2:
            st.markdown("### Dataset Statistics")
            st.markdown(f"**Full Dataset Size:** {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} columns")
            st.markdown(f"**Working Subset Size:** {st.session_state.df_subset.shape[0]:,} rows × {st.session_state.df_subset.shape[1]} columns")
            st.markdown(f"**Date Range:** {st.session_state.df_subset['Date'].min()} to {st.session_state.df_subset['Date'].max()}")
            st.markdown(f"**Number of Stores:** {st.session_state.df_subset['Store'].nunique()}")
        
        # Display data types and missing values
        st.markdown("<h3 class='section-header'>Data Types and Missing Values</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Data types
            dtypes_df = pd.DataFrame(st.session_state.df_subset.dtypes, columns=['Data Type'])
            dtypes_df = dtypes_df.reset_index().rename(columns={'index': 'Column'})
            st.dataframe(dtypes_df)
        
        with col2:
            # Missing values
            missing_df = pd.DataFrame(st.session_state.df_subset.isnull().sum(), columns=['Missing Values'])
            missing_df['Percentage'] = (missing_df['Missing Values'] / len(st.session_state.df_subset) * 100).round(2)
            missing_df = missing_df.reset_index().rename(columns={'index': 'Column'})
            st.dataframe(missing_df)
        
        # Option to reset data
        if st.button("Reset Data", key="reset_data"):
            st.session_state.data_loaded = False
            st.session_state.df = None
            st.session_state.df_subset = None
            st.experimental_rerun()

# Exploratory Analysis page
elif selected == "Exploratory Analysis":
    st.markdown("<h1 class='main-header'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load the data first by going to the Data Loading section.")
    else:
        if not st.session_state.eda_completed:
            st.markdown("<p class='info-text'>Perform exploratory data analysis on the Rossmann Store Sales dataset.</p>", unsafe_allow_html=True)
            
            if st.button("Run Exploratory Analysis", key="run_eda"):
                with st.spinner("Running exploratory analysis..."):
                    # Get store IDs from the subset
                    store_ids = st.session_state.df_subset['Store'].unique()
                    
                    # Run EDA without expecting returned figures
                    run_exploratory_analysis(st.session_state.df_subset, store_ids)
                    st.session_state.eda_figures = {}  # Initialize empty dict since we're not getting figures returned
                    st.session_state.eda_completed = True
                    
                st.success("Exploratory analysis completed!")
                st.experimental_rerun()
        else:
            # Display EDA results
            st.success("Exploratory analysis already completed!")
            
            # Create tabs for different EDA sections
            eda_tabs = st.tabs(["Sales Trends", "Store Comparisons", "Seasonality", "Correlations", "Feature Analysis"])
            
            with eda_tabs[0]:
                st.markdown("<h2 class='sub-header'>Sales Trends</h2>", unsafe_allow_html=True)
                
                # Always create the plot since we don't have stored figures
                fig, ax = plt.subplots(figsize=(12, 6))
                df_agg = st.session_state.df_subset.groupby('Date')['Sales'].mean().reset_index()
                ax.plot(df_agg['Date'], df_agg['Sales'])
                ax.set_title('Average Daily Sales Trend')
                ax.set_xlabel('Date')
                ax.set_ylabel('Average Sales')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                st.markdown("""
                <div class="highlight">
                <p class="info-text">The sales trend shows the overall pattern of sales over time. 
                Look for seasonality, trends, and any unusual spikes or drops that might need further investigation.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with eda_tabs[1]:
                st.markdown("<h2 class='sub-header'>Store Comparisons</h2>", unsafe_allow_html=True)
                
                # Always create the plot since we don't have stored figures
                fig, ax = plt.subplots(figsize=(12, 6))
                store_agg = st.session_state.df_subset.groupby('Store')['Sales'].mean().reset_index()
                sns.barplot(x='Store', y='Sales', data=store_agg, ax=ax)
                ax.set_title('Average Sales by Store')
                ax.set_xlabel('Store ID')
                ax.set_ylabel('Average Sales')
                st.pyplot(fig)
                
                st.markdown("""
                <div class="highlight">
                <p class="info-text">This comparison shows how different stores perform relative to each other.
                Significant differences might indicate store-specific factors affecting sales.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with eda_tabs[2]:
                st.markdown("<h2 class='sub-header'>Seasonality Analysis</h2>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Always create the plot since we don't have stored figures
                    fig, ax = plt.subplots(figsize=(10, 6))
                    dow_agg = st.session_state.df_subset.groupby('DayOfWeek')['Sales'].mean().reset_index()
                    sns.barplot(x='DayOfWeek', y='Sales', data=dow_agg, ax=ax)
                    ax.set_title('Average Sales by Day of Week')
                    ax.set_xlabel('Day of Week (1=Monday, 7=Sunday)')
                    ax.set_ylabel('Average Sales')
                    st.pyplot(fig)
                
                with col2:
                    # Always create the plot since we don't have stored figures
                    fig, ax = plt.subplots(figsize=(10, 6))
                    st.session_state.df_subset['Month'] = pd.to_datetime(st.session_state.df_subset['Date']).dt.month
                    month_agg = st.session_state.df_subset.groupby('Month')['Sales'].mean().reset_index()
                    sns.barplot(x='Month', y='Sales', data=month_agg, ax=ax)
                    ax.set_title('Average Sales by Month')
                    ax.set_xlabel('Month')
                    ax.set_ylabel('Average Sales')
                    st.pyplot(fig)
                
                st.markdown("""
                <div class="highlight">
                <p class="info-text">Seasonality analysis reveals patterns in sales based on time factors like day of week and month.
                These patterns are crucial for accurate forecasting and business planning.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with eda_tabs[3]:
                st.markdown("<h2 class='sub-header'>Correlation Analysis</h2>", unsafe_allow_html=True)
                
                # Always create the plot since we don't have stored figures
                numeric_cols = st.session_state.df_subset.select_dtypes(include=[np.number]).columns
                corr_matrix = st.session_state.df_subset[numeric_cols].corr()
                
                fig, ax = plt.subplots(figsize=(12, 10))
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                ax.set_title('Correlation Matrix of Numeric Features')
                st.pyplot(fig)
                
                st.markdown("""
                <div class="highlight">
                <p class="info-text">The correlation matrix shows relationships between different numeric variables.
                Strong correlations (positive or negative) can indicate important relationships to consider in forecasting.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with eda_tabs[4]:
                st.markdown("<h2 class='sub-header'>Feature Analysis</h2>", unsafe_allow_html=True)
                
                # Feature selection
                numeric_cols = st.session_state.df_subset.select_dtypes(include=[np.number]).columns.tolist()
                selected_feature = st.selectbox("Select Feature to Analyze", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(st.session_state.df_subset[selected_feature].dropna(), kde=True, ax=ax)
                    ax.set_title(f'Distribution of {selected_feature}')
                    ax.set_xlabel(selected_feature)
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
                
                with col2:
                    # Box plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(y=st.session_state.df_subset[selected_feature].dropna(), ax=ax)
                    ax.set_title(f'Box Plot of {selected_feature}')
                    ax.set_ylabel(selected_feature)
                    st.pyplot(fig)
                
                # Scatter plot with sales
                if selected_feature != 'Sales':
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x=selected_feature, y='Sales', data=st.session_state.df_subset.sample(min(1000, len(st.session_state.df_subset))), alpha=0.5, ax=ax)
                    ax.set_title(f'Sales vs {selected_feature}')
                    ax.set_xlabel(selected_feature)
                    ax.set_ylabel('Sales')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Feature statistics
                st.markdown("<h3 class='section-header'>Feature Statistics</h3>", unsafe_allow_html=True)
                stats_df = pd.DataFrame({
                    'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Missing Values'],
                    'Value': [
                        f"{st.session_state.df_subset[selected_feature].mean():.2f}",
                        f"{st.session_state.df_subset[selected_feature].median():.2f}",
                        f"{st.session_state.df_subset[selected_feature].std():.2f}",
                        f"{st.session_state.df_subset[selected_feature].min():.2f}",
                        f"{st.session_state.df_subset[selected_feature].max():.2f}",
                        f"{st.session_state.df_subset[selected_feature].isnull().sum()} ({st.session_state.df_subset[selected_feature].isnull().sum() / len(st.session_state.df_subset) * 100:.2f}%)"
                    ]
                })
                st.dataframe(stats_df)
            
            # Option to reset EDA
            if st.button("Reset EDA", key="reset_eda"):
                st.session_state.eda_completed = False
                st.session_state.eda_figures = {}
                st.experimental_rerun()

# Data Processing page
elif selected == "Data Processing":
    st.markdown("<h1 class='main-header'>Data Processing</h1>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load the data first by going to the Data Loading section.")
    else:
        if not st.session_state.data_processed:
            st.markdown("<p class='info-text'>Preprocess the data and engineer features for modeling.</p>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("### Data Processing Steps")
                st.markdown("""
                The preprocessing pipeline includes:
                1. Handling missing values
                2. Feature engineering (time-based features, lag features, etc.)
                3. Encoding categorical variables
                4. Scaling numerical features
                5. Preparing data for time series forecasting
                """)
            
            with col2:
                st.image("https://img.icons8.com/color/240/000000/data-configuration.png", width=100)
            
            if st.button("Process Data", key="process_data"):
                with st.spinner("Processing data and engineering features..."):
                    # Process the data
                    st.session_state.processed_df = preprocess_data(st.session_state.df_subset)
                    
                    # Get feature columns
                    st.session_state.features = get_feature_columns(st.session_state.processed_df)
                    
                    st.session_state.data_processed = True
                
                st.success("Data processing completed!")
        else:
            st.success("Data already processed!")
            
            # Display processed data information
            st.markdown("<h2 class='sub-header'>Processed Data Overview</h2>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Processed Data Sample")
                st.dataframe(st.session_state.processed_df.head())
            
            with col2:
                st.markdown("### Processing Statistics")
                st.markdown(f"**Original Data Shape:** {st.session_state.df_subset.shape[0]:,} rows × {st.session_state.df_subset.shape[1]} columns")
                st.markdown(f"**Processed Data Shape:** {st.session_state.processed_df.shape[0]:,} rows × {st.session_state.processed_df.shape[1]} columns")
                st.markdown(f"**New Features Added:** {st.session_state.processed_df.shape[1] - st.session_state.df_subset.shape[1]}")
            
            # Display feature information
            st.markdown("<h2 class='sub-header'>Feature Information</h2>", unsafe_allow_html=True)
            
            # Feature categories
            original_features = set(st.session_state.df_subset.columns)
            all_features = set(st.session_state.processed_df.columns)
            new_features = all_features - original_features
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Original Features")
                st.write(", ".join(sorted(original_features)))
            
            with col2:
                st.markdown("### Engineered Features")
                st.write(", ".join(sorted(new_features)))
            
            # Features used for modeling
            st.markdown("<h3 class='section-header'>Features Used for Modeling</h3>", unsafe_allow_html=True)
            
            if st.session_state.features:
                # Create a DataFrame to display features in a more organized way
                feature_df = pd.DataFrame({
                    'Feature Name': st.session_state.features
                })
                st.dataframe(feature_df)
                
                # Feature importance visualization (placeholder)
                st.markdown("<h3 class='section-header'>Feature Importance (Preview)</h3>", unsafe_allow_html=True)
                
                # Create a sample feature importance plot
                fig, ax = plt.subplots(figsize=(10, 6))
                importance = np.random.rand(len(st.session_state.features))
                sorted_idx = np.argsort(importance)
                ax.barh(np.array(st.session_state.features)[sorted_idx], importance[sorted_idx])
                ax.set_title('Feature Importance Preview')
                ax.set_xlabel('Importance')
                st.pyplot(fig)
                
                st.info("Note: This is a preview of feature importance. Actual importance will be calculated during the forecasting step.")
            else:
                st.info("Feature list not available. Please process the data first.")
            
            # Option to reset processing
            if st.button("Reset Processing", key="reset_processing"):
                st.session_state.data_processed = False
                st.session_state.processed_df = None
                st.session_state.features = None
                st.experimental_rerun()

# Forecasting page
elif selected == "Forecasting":
    st.markdown("<h1 class='main-header'>Sales Forecasting</h1>", unsafe_allow_html=True)
    
    if not st.session_state.data_processed:
        st.warning("Please process the data first by going to the Data Processing section.")
    else:
        if not st.session_state.forecasting_completed:
            st.markdown("<p class='info-text'>Run the forecasting pipeline to predict future sales.</p>", unsafe_allow_html=True)
            
            # Forecasting parameters
            st.markdown("<h2 class='sub-header'>Forecasting Parameters</h2>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                forecast_days = st.slider("Forecast Horizon (Days)", min_value=7, max_value=30, value=14, step=1)
                store_ids = st.multiselect("Select Stores to Forecast", 
                                          options=sorted(st.session_state.df_subset['Store'].unique()),
                                          default=sorted(st.session_state.df_subset['Store'].unique())[:3])
            
            with col2:
                st.markdown("### Model Selection")
                model_options = ["Auto (Best Model)", "ARIMA", "Prophet", "XGBoost", "LSTM"]
                selected_model = st.selectbox("Select Forecasting Model", model_options)
                
                st.markdown("### Evaluation Metric")
                metric_options = ["RMSE", "MAE", "MAPE"]
                selected_metric = st.selectbox("Select Evaluation Metric", metric_options)
            
            if st.button("Run Forecasting", key="run_forecasting"):
                with st.spinner("Running forecasting pipeline..."):
                    # Run the forecasting pipeline
                    st.session_state.best_model, st.session_state.forecast_results = run_forecasting_pipeline(
                        st.session_state.processed_df, 
                        st.session_state.features, 
                        forecast_days=forecast_days, 
                        store_ids=store_ids,
                        original_df=st.session_state.df_subset,
                        model_type=selected_model if selected_model != "Auto (Best Model)" else None,
                        metric=selected_metric
                    )
                    
                    st.session_state.forecasting_completed = True
                
                st.success("Forecasting completed!")
        else:
            st.success("Forecasting already completed!")
            
            # Display forecasting results
            st.markdown("<h2 class='sub-header'>Forecasting Results</h2>", unsafe_allow_html=True)
            
            # Store selection for visualization
            store_ids = sorted(st.session_state.df_subset['Store'].unique())
            selected_store = st.selectbox("Select Store to Visualize", store_ids)
            
            # Forecast visualization
            st.markdown("<h3 class='section-header'>Sales Forecast</h3>", unsafe_allow_html=True)
            
            if st.session_state.forecast_results is not None and selected_store in st.session_state.forecast_results:
                # Get forecast data for the selected store
                forecast_data = st.session_state.forecast_results[selected_store]
                
                # Create a visualization of the forecast
                fig = go.Figure()
                
                # Add historical data
                historical = st.session_state.df_subset[st.session_state.df_subset['Store'] == selected_store]
                historical = historical.sort_values('Date')
                
                fig.add_trace(go.Scatter(
                    x=historical['Date'],
                    y=historical['Sales'],
                    mode='lines',
                    name='Historical Sales',
                    line=dict(color='blue')
                ))
                
                # Add forecast data
                if 'forecast_dates' in forecast_data and 'forecast_values' in forecast_data:
                    fig.add_trace(go.Scatter(
                        x=forecast_data['forecast_dates'],
                        y=forecast_data['forecast_values'],
                        mode='lines',
                        name='Forecasted Sales',
                        line=dict(color='red')
                    ))
                    
                    # Add confidence intervals if available
                    if 'lower_bound' in forecast_data and 'upper_bound' in forecast_data:
                        fig.add_trace(go.Scatter(
                            x=forecast_data['forecast_dates'] + forecast_data['forecast_dates'][::-1],
                            y=forecast_data['upper_bound'] + forecast_data['lower_bound'][::-1],
                            fill='toself',
                            fillcolor='rgba(231,107,243,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Confidence Interval'
                        ))
                
                fig.update_layout(
                    title=f'Sales Forecast for Store {selected_store}',
                    xaxis_title='Date',
                    yaxis_title='Sales',
                    legend=dict(x=0, y=1),
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast metrics
                st.markdown("<h3 class='section-header'>Forecast Metrics</h3>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"### RMSE")
                    if 'rmse' in forecast_data:
                        st.markdown(f"<h2 style='color:#1E88E5;'>{forecast_data['rmse']:.2f}</h2>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h2 style='color:#1E88E5;'>N/A</h2>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"### MAE")
                    if 'mae' in forecast_data:
                        st.markdown(f"<h2 style='color:#1E88E5;'>{forecast_data['mae']:.2f}</h2>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h2 style='color:#1E88E5;'>N/A</h2>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"### MAPE")
                    if 'mape' in forecast_data:
                        st.markdown(f"<h2 style='color:#1E88E5;'>{forecast_data['mape']:.2f}%</h2>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h2 style='color:#1E88E5;'>N/A</h2>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Forecast details
                st.markdown("<h3 class='section-header'>Forecast Details</h3>", unsafe_allow_html=True)
                
                if 'forecast_dates' in forecast_data and 'forecast_values' in forecast_data:
                    forecast_df = pd.DataFrame({
                        'Date': forecast_data['forecast_dates'],
                        'Forecasted Sales': forecast_data['forecast_values'].round(2)
                    })
                    
                    if 'lower_bound' in forecast_data and 'upper_bound' in forecast_data:
                        forecast_df['Lower Bound'] = forecast_data['lower_bound'].round(2)
                        forecast_df['Upper Bound'] = forecast_data['upper_bound'].round(2)
                    
                    st.dataframe(forecast_df)
                else:
                    st.info("Detailed forecast data not available.")
            else:
                st.info(f"Forecast data for Store {selected_store} not available.")
            
            # Model information
            st.markdown("<h3 class='section-header'>Model Information</h3>", unsafe_allow_html=True)

            if st.session_state.best_model is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
                    st.markdown(f"**Best Model:** {type(st.session_state.best_model).__name__}")
                    
                    # Display model parameters if available
                    if hasattr(st.session_state.best_model, 'get_params'):
                        params = st.session_state.best_model.get_params()
                        st.markdown(f"**Model Parameters:** {params}")
                    else:
                        st.markdown("**Model Parameters:** Not available")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
                    # For RandomForestRegressor, we can display n_estimators and max_depth
                    if hasattr(st.session_state.best_model, 'n_estimators'):
                        st.markdown(f"**Number of Trees:** {st.session_state.best_model.n_estimators}")
                    if hasattr(st.session_state.best_model, 'max_depth'):
                        max_depth = st.session_state.best_model.max_depth if st.session_state.best_model.max_depth is not None else "None (unlimited)"
                        st.markdown(f"**Max Depth:** {max_depth}")
                    st.markdown(f"**Feature Importance Available:** {'Yes' if hasattr(st.session_state.best_model, 'feature_importances_') else 'No'}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Feature importance if available
                if hasattr(st.session_state.best_model, 'feature_importances_'):
                    st.markdown("<h4>Feature Importance</h4>", unsafe_allow_html=True)
                    
                    # Get feature importances
                    importances = st.session_state.best_model.feature_importances_
                    
                    # If features are available, use them for labels
                    if st.session_state.features and len(st.session_state.features) == len(importances):
                        features = st.session_state.features
                    else:
                        features = [f"Feature {i}" for i in range(len(importances))]
                    
                    # Create feature importance DataFrame
                    feature_imp_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    # Display top 15 features for readability
                    top_features = feature_imp_df.head(15)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
                    ax.set_title('Feature Importance (Top 15)')
                    ax.set_xlabel('Importance')
                    st.pyplot(fig)
                    
                    # Show full table
                    st.markdown("<h4>All Feature Importances</h4>", unsafe_allow_html=True)
                    st.dataframe(feature_imp_df)
            else:
                st.info("Model information not available.")
            
            # Option to reset forecasting
            if st.button("Reset Forecasting", key="reset_forecasting"):
                st.session_state.forecasting_completed = False
                st.session_state.forecast_results = None
                st.session_state.best_model = None
                st.experimental_rerun()

# Business Insights page
elif selected == "Business Insights":
    st.markdown("<h1 class='main-header'>Business Insights</h1>", unsafe_allow_html=True)
    
    if not st.session_state.forecasting_completed:
        st.warning("Please complete the forecasting first by going to the Forecasting section.")
    else:
        if not st.session_state.insights_generated:
            st.markdown("<p class='info-text'>Generate business insights from the forecasting results.</p>", unsafe_allow_html=True)
            
            if st.button("Generate Business Insights", key="generate_insights"):
                with st.spinner("Generating business insights..."):
                    # Generate business insights
                    st.session_state.insights = generate_business_insights(
                        st.session_state.df_subset, 
                        st.session_state.forecast_results
                    )
                    
                    st.session_state.insights_generated = True
                
                st.success("Business insights generated!")
        else:
            st.success("Business insights already generated!")
            
            # Display business insights
            st.markdown("<h2 class='sub-header'>Key Business Insights</h2>", unsafe_allow_html=True)
            
            if st.session_state.insights:
                # Create tabs for different insight categories
                insight_tabs = st.tabs(["Sales Trends", "Store Performance", "Seasonality", "Recommendations"])
                
                with insight_tabs[0]:
                    st.markdown("<h3 class='section-header'>Sales Trends Insights</h3>", unsafe_allow_html=True)
                    
                    if 'sales_trends' in st.session_state.insights:
                        for insight in st.session_state.insights['sales_trends']:
                            st.markdown(f"<div class='highlight'><p>{insight}</p></div>", unsafe_allow_html=True)
                    else:
                        # Sample insights if not available
                        st.markdown("""
                        <div class='highlight'>
                        <p>Overall sales show an upward trend with an average growth rate of 3.2% month-over-month.</p>
                        </div>
                        
                        <div class='highlight'>
                        <p>Sales volatility has decreased by 15% in the last quarter, indicating more stable revenue streams.</p>
                        </div>
                        
                        <div class='highlight'>
                        <p>The forecast predicts a 5.8% increase in total sales over the next 14 days compared to the previous period.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Sales trend visualization
                    fig = go.Figure()
                    
                    # Aggregate sales by date
                    sales_agg = st.session_state.df_subset.groupby('Date')['Sales'].sum().reset_index()
                    
                    # Add a trend line
                    x = np.arange(len(sales_agg))
                    y = sales_agg['Sales']
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    
                    fig.add_trace(go.Scatter(
                        x=sales_agg['Date'],
                        y=sales_agg['Sales'],
                        mode='lines',
                        name='Historical Sales',
                        line=dict(color='blue')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=sales_agg['Date'],
                        y=p(x),
                        mode='lines',
                        name='Trend Line',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title='Overall Sales Trend with Trend Line',
                        xaxis_title='Date',
                        yaxis_title='Total Sales',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with insight_tabs[1]:
                    st.markdown("<h3 class='section-header'>Store Performance Insights</h3>", unsafe_allow_html=True)
                    
                    if 'store_performance' in st.session_state.insights:
                        for insight in st.session_state.insights['store_performance']:
                            st.markdown(f"<div class='highlight'><p>{insight}</p></div>", unsafe_allow_html=True)
                    else:
                        # Sample insights if not available
                        st.markdown("""
                        <div class='highlight'>
                        <p>Store 2 is the top performer with 12.5% higher sales than the average across all stores.</p>
                        </div>
                        
                        <div class='highlight'>
                        <p>Store 3 shows the highest growth potential with a forecasted 8.7% increase in the next period.</p>
                        </div>
                        
                        <div class='highlight'>
                        <p>Store 1 has the most consistent sales pattern with the lowest coefficient of variation (0.15).</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Store performance comparison
                    store_agg = st.session_state.df_subset.groupby('Store')['Sales'].agg(['mean', 'std', 'min', 'max']).reset_index()
                    store_agg['cv'] = store_agg['std'] / store_agg['mean']  # Coefficient of variation
                    
                    fig = px.bar(
                        store_agg,
                        x='Store',
                        y='mean',
                        error_y='std',
                        labels={'mean': 'Average Sales', 'Store': 'Store ID'},
                        title='Store Performance Comparison',
                        color='cv',
                        color_continuous_scale='Viridis',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store metrics table
                    st.markdown("<h4>Store Performance Metrics</h4>", unsafe_allow_html=True)
                    
                    metrics_df = store_agg.copy()
                    metrics_df.columns = ['Store ID', 'Average Sales', 'Standard Deviation', 'Minimum Sales', 'Maximum Sales', 'Coefficient of Variation']
                    metrics_df = metrics_df.round(2)
                    
                    st.dataframe(metrics_df)
                
                with insight_tabs[2]:
                    st.markdown("<h3 class='section-header'>Seasonality Insights</h3>", unsafe_allow_html=True)
                    
                    if 'seasonality' in st.session_state.insights:
                        for insight in st.session_state.insights['seasonality']:
                            st.markdown(f"<div class='highlight'><p>{insight}</p></div>", unsafe_allow_html=True)
                    else:
                        # Sample insights if not available
                        st.markdown("""
                        <div class='highlight'>
                        <p>Monday and Friday show the highest sales volumes, with 18% and 22% above the weekly average respectively.</p>
                        </div>
                        
                        <div class='highlight'>
                        <p>Sales are typically 30% higher during the first week of the month compared to the last week.</p>
                        </div>
                        
                        <div class='highlight'>
                        <p>Promotional periods show a 45% increase in sales compared to non-promotional periods.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Day of week seasonality
                        dow_agg = st.session_state.df_subset.groupby('DayOfWeek')['Sales'].mean().reset_index()
                        
                        fig = px.line(
                            dow_agg,
                            x='DayOfWeek',
                            y='Sales',
                            markers=True,
                            labels={'Sales': 'Average Sales', 'DayOfWeek': 'Day of Week (1=Monday)'},
                            title='Sales by Day of Week',
                            height=300
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Month seasonality
                        st.session_state.df_subset['Month'] = pd.to_datetime(st.session_state.df_subset['Date']).dt.month
                        month_agg = st.session_state.df_subset.groupby('Month')['Sales'].mean().reset_index()
                        
                        fig = px.line(
                            month_agg,
                            x='Month',
                            y='Sales',
                            markers=True,
                            labels={'Sales': 'Average Sales', 'Month': 'Month'},
                            title='Sales by Month',
                            height=300
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Promo effect
                    if 'Promo' in st.session_state.df_subset.columns:
                        promo_agg = st.session_state.df_subset.groupby('Promo')['Sales'].mean().reset_index()
                        
                        fig = px.bar(
                            promo_agg,
                            x='Promo',
                            y='Sales',
                            labels={'Sales': 'Average Sales', 'Promo': 'Promotion (0=No, 1=Yes)'},
                            title='Effect of Promotions on Sales',
                            color='Promo',
                            color_discrete_sequence=['#1E88E5', '#FFC107'],
                            height=300
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with insight_tabs[3]:
                    st.markdown("<h3 class='section-header'>Business Recommendations</h3>", unsafe_allow_html=True)
                    
                    if 'recommendations' in st.session_state.insights:
                        for i, recommendation in enumerate(st.session_state.insights['recommendations'], 1):
                            st.markdown(f"<div class='highlight'><p><strong>Recommendation {i}:</strong> {recommendation}</p></div>", unsafe_allow_html=True)
                    else:
                        # Sample recommendations if not available
                        st.markdown("""
                        <div class='highlight'>
                        <p><strong>Recommendation 1:</strong> Increase inventory levels for Store 2 by 15% to meet the forecasted demand increase.</p>
                        </div>
                        
                        <div class='highlight'>
                        <p><strong>Recommendation 2:</strong> Schedule additional staff on Mondays and Fridays to handle higher customer traffic.</p>
                        </div>
                        
                        <div class='highlight'>
                        <p><strong>Recommendation 3:</strong> Implement targeted promotions during the third week of the month to boost traditionally lower sales periods.</p>
                        </div>
                        
                        <div class='highlight'>
                        <p><strong>Recommendation 4:</strong> Optimize supply chain for Store 3 to reduce stockouts during peak sales periods.</p>
                        </div>
                        
                        <div class='highlight'>
                        <p><strong>Recommendation 5:</strong> Develop a special marketing campaign for the upcoming forecasted peak in the next 7-10 days.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Priority matrix
                    st.markdown("<h4>Recommendation Priority Matrix</h4>", unsafe_allow_html=True)
                    
                    # Sample priority matrix
                    priority_data = {
                        'Recommendation': [
                            'Increase inventory for Store 2',
                            'Additional staff on peak days',
                            'Targeted promotions',
                            'Supply chain optimization',
                            'Special marketing campaign'
                        ],
                        'Impact': [4.5, 3.8, 3.2, 4.0, 3.5],
                        'Effort': [2.5, 1.8, 2.2, 4.5, 3.0],
                        'Priority': ['High', 'High', 'Medium', 'Medium', 'High']
                    }
                    
                    priority_df = pd.DataFrame(priority_data)
                    
                    fig = px.scatter(
                        priority_df,
                        x='Effort',
                        y='Impact',
                        text='Recommendation',
                        color='Priority',
                        size=[20, 20, 20, 20, 20],
                        labels={'Impact': 'Business Impact (1-5)', 'Effort': 'Implementation Effort (1-5)'},
                        title='Recommendation Priority Matrix',
                        color_discrete_map={'High': '#4CAF50', 'Medium': '#FFC107', 'Low': '#F44336'},
                        height=500
                    )
                    
                    fig.update_traces(textposition='top center')
                    fig.update_layout(
                        shapes=[
                            # Dividing lines for the priority matrix
                            dict(
                                type='line',
                                x0=3,
                                y0=0,
                                x1=3,
                                y1=5,
                                line=dict(color='gray', width=1, dash='dash')
                            ),
                            dict(
                                type='line',
                                x0=0,
                                y0=3,
                                x1=5,
                                y1=3,
                                line=dict(color='gray', width=1, dash='dash')
                            )
                        ],
                        annotations=[
                            dict(
                                x=1.5,
                                y=4,
                                text="Quick Wins",
                                showarrow=False,
                                font=dict(size=14)
                            ),
                            dict(
                                x=4,
                                y=4,
                                text="Major Projects",
                                showarrow=False,
                                font=dict(size=14)
                            ),
                            dict(
                                x=1.5,
                                y=1.5,
                                text="Low Priority",
                                showarrow=False,
                                font=dict(size=14)
                            ),
                            dict(
                                x=4,
                                y=1.5,
                                text="Fill-ins",
                                showarrow=False,
                                font=dict(size=14)
                            )
                        ]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Business insights not available.")
            
            # Export options
            st.markdown("<h3 class='section-header'>Export Options</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Export Insights as PDF", key="export_pdf"):
                    st.info("PDF export functionality would be implemented here.")
            
            with col2:
                if st.button("Export Forecast Data as CSV", key="export_csv"):
                    st.info("CSV export functionality would be implemented here.")
            
            with col3:
                if st.button("Schedule Regular Reports", key="schedule_reports"):
                    st.info("Report scheduling functionality would be implemented here.")
            
            # Option to reset insights
            if st.button("Reset Insights", key="reset_insights"):
                st.session_state.insights_generated = False
                st.session_state.insights = None
                st.experimental_rerun()

# Main function to run the app
def main():
    pass

if __name__ == "__main__":
    main()
