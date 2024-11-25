import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pydeck as pdk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Ensure 'logged_in' is initialized in session_state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Title is always displayed at the beginning
st.title("Project Dashboard")

# Check if the user is logged in
if not st.session_state.logged_in:
    # Show login prompt only before login
    st.markdown("Please log in to see the dashboard content.")

    # Login form
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    login_button_clicked = st.sidebar.button("Login")  # Track button click

    if login_button_clicked:
        # Check the credentials and update session state immediately
        if username == "group6" and password == "group6":
            st.session_state.logged_in = True
            st.sidebar.success("Login successful!")
            st.rerun()  # Use st.rerun() to re-run the app to update UI
        else:
            st.sidebar.error("Invalid username or password.")
else:
    # After login: Show the actual dashboard content
    st.sidebar.success("Logged in as group6")
    st.markdown("This dashboard presents detailed insights on the dataset.")

    # Define dataset paths
    combined_dataset_path = "DataSetFolder/Amazon product/P1-AmazingMartEU2.xlsx"
    csv_dataset_path = "DataSetFolder/Amazon Sales FY2020-21/Amazon Sales FY2020-21.csv"
    product_folder_path = "DataSetFolder/Amazon Products Dataset"
    preprocessed_dataset_path = "data/preprocessed_combined_dataset.csv"

    # Cache for loading datasets
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_data():
        product_excel_df = pd.read_excel(combined_dataset_path)
        product_csv_df = pd.read_csv(csv_dataset_path, low_memory=False)
        csv_files = [f for f in os.listdir(product_folder_path) if f.endswith('.csv')]
        product_folder_dfs = [pd.read_csv(os.path.join(product_folder_path, f)) for f in csv_files]
        combined_data = pd.concat([product_excel_df, product_csv_df] + product_folder_dfs, ignore_index=True)
        return combined_data

    # Cache for data preprocessing
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def clean_and_process_data(data):
        data['price'] = pd.to_numeric(data['price'].astype(str).str.replace('₹', '').str.replace(',', ''), errors='coerce')
        data['order_date'] = pd.to_datetime(data['order_date'], errors='coerce')
        data['year_month'] = data['order_date'].dt.to_period('M').astype(str)
        data['month'] = data['order_date'].dt.month
        data['total'] = data['price'] * data['qty_ordered']
        data = data.dropna(subset=['price', 'qty_ordered', 'order_date'])  # Ensure no missing critical values
        return data

    # Cache for loading preprocessed data
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_preprocessed_data():
        return pd.read_csv(preprocessed_dataset_path)

    try:
        # Load datasets
        raw_data = load_data()
        data_cleaned = clean_and_process_data(raw_data)

        # Load preprocessed data
        data = load_preprocessed_data()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")

    # Sidebar Navigation for Visualizations
    st.sidebar.title("Choose Visualization")
    visualization = st.sidebar.selectbox(
        "Select the insight you want to explore:",
        [
            "Dataset Overview",
            "Best Performing Category",
            "Purchase Difference Between Years",
            "Purchase Analysis by Gender",
            "Purchase Analysis by Age Group",
            "3D Map of Purchases by State",
            "EDA",
            "Model Development",
        ],
    )

    # Add logic for visualizations here
    if visualization == "Dataset Overview":
        st.write("### Dataset Overview")
        st.dataframe(data.head())

        st.write("#### Summary Statistics")
        st.write(data.describe())

        st.write("#### Missing Values")
        st.write(data.isnull().sum())

    # Best Performing Category
    if visualization == "Best Performing Category":
        st.write("### Best Performing Category")
        best_category = data.groupby("category")["total"].sum().reset_index()
        best_category = best_category.sort_values(by="total", ascending=False)

        chart_type = st.selectbox("Select chart type:", ["Bar Chart", "Line Graph"])
        if chart_type == "Bar Chart":
            st.bar_chart(best_category.set_index("category"))
        elif chart_type == "Line Graph":
            st.line_chart(best_category.set_index("category"))

        st.write("#### Best Performing Category Per Year")
        best_per_year = (
            data.groupby(["year", "category"])["total"].sum().reset_index()
            .sort_values(["year", "total"], ascending=[True, False])
            .groupby("year").first().reset_index()
        )
        st.dataframe(best_per_year.style.format({"total": "{:,.2f}"}))

    # Purchase Difference Between Years
    if visualization == "Purchase Difference Between Years":
        st.write("### Purchase Difference Between Years")
        year_comparison = data.groupby("year")["total"].sum().reset_index()
        st.line_chart(year_comparison.set_index("year"))

    # Purchase Analysis by Gender
    if visualization == "Purchase Analysis by Gender":
        st.write("### Purchase Analysis by Gender")
        gender_analysis = data.groupby("Gender")["total"].sum().reset_index()
        fig_gender = px.pie(
            gender_analysis,
            names="Gender",
            values="total",
            title="Purchases by Gender",
        )
        st.plotly_chart(fig_gender)

        st.write("#### Top 3 Categories per Gender")
        gender_category_top = data.groupby(["Gender", "category"])["total"].sum().reset_index()
        gender_category_top = gender_category_top.sort_values(by=["Gender", "total"], ascending=[True, False])
        top_gender_categories = gender_category_top.groupby("Gender").head(3)
        st.table(top_gender_categories)

    # Purchase Analysis by Age Group
    if visualization == "Purchase Analysis by Age Group":
        st.write("### Purchase Analysis by Age Group")
        bins = [0, 18, 25, 35, 50, 65, 100]
        labels = ["<18", "18-25", "26-35", "36-50", "51-65", "65+"]
        data["Age Group"] = pd.cut(data["age"], bins=bins, labels=labels, right=False)
        age_group_analysis = data.groupby("Age Group")["total"].sum().reset_index()
        fig_age = px.bar(
            age_group_analysis,
            x="Age Group",
            y="total",
            title="Purchases by Age Group",
        )
        st.plotly_chart(fig_age)

        st.write("#### Top 3 Categories per Age Group")
        age_group_category_top = data.groupby(["Age Group", "category"])["total"].sum().reset_index()
        age_group_category_top = age_group_category_top.sort_values(by=["Age Group", "total"], ascending=[True, False])
        top_age_group_categories = age_group_category_top.groupby("Age Group").head(3)
        st.table(top_age_group_categories)

    if visualization == "3D Map of Purchases by State":
        st.write("### 3D Map of Purchases by State")
        
        # State coordinates dictionary
        state_coordinates = {
            "AL": [32.806671, -86.791130], "AK": [61.370716, -152.404419],
            "AZ": [33.729759, -111.431221], "AR": [34.969704, -92.373123],
            "CA": [36.116203, -119.681564], "CO": [39.059811, -105.311104],
            "CT": [41.597782, -72.755371], "DE": [39.318523, -75.507141],
            "FL": [27.766279, -81.686783], "GA": [33.040619, -83.643074],
            "HI": [21.094318, -157.498337], "ID": [44.240459, -114.478828],
            "IL": [40.349457, -88.986137], "IN": [39.849426, -86.258278],
            "IA": [42.011539, -93.210526], "KS": [38.526600, -96.726486],
            "KY": [37.668140, -84.670067], "LA": [31.169546, -91.867805],
            "ME": [44.693947, -69.381927], "MD": [39.063946, -76.802101],
            "MA": [42.230171, -71.530106], "MI": [43.326618, -84.536095],
            "MN": [45.694454, -93.900192], "MS": [32.741646, -89.678696],
            "MO": [38.456085, -92.288368], "MT": [46.921925, -110.454353],
            "NE": [41.125370, -98.268082], "NV": [38.313515, -117.055374],
            "NH": [43.452492, -71.563896], "NJ": [40.298904, -74.521011],
            "NM": [34.840515, -106.248482], "NY": [42.165726, -74.948051],
            "NC": [35.630066, -79.806419], "ND": [47.528912, -99.784012],
            "OH": [40.388783, -82.764915], "OK": [35.565342, -96.928917],
            "OR": [44.572021, -122.070938], "PA": [40.590752, -77.209755],
            "RI": [41.680893, -71.511780], "SC": [33.856892, -80.945007],
            "SD": [44.299782, -99.438828], "TN": [35.747845, -86.692345],
            "TX": [31.054487, -97.563461], "UT": [40.150032, -111.862434],
            "VT": [44.045876, -72.710686], "VA": [37.769337, -78.169968],
            "WA": [47.400902, -121.490494], "WV": [38.491226, -80.954456],
            "WI": [44.268543, -89.616508], "WY": [42.755966, -107.302490]
        }
    
        # Aggregating state data
        state_data = data.groupby("State_x").agg(
            total_sales=("total", "sum"),
            total_buyers=("cust_id", "nunique"),
            top_category=("category", lambda x: x.mode()[0]),
        ).reset_index()
    
        # Adding latitude and longitude based on state
        state_data["lat"] = state_data["State_x"].map(lambda x: state_coordinates.get(x, [0, 0])[0])
        state_data["lon"] = state_data["State_x"].map(lambda x: state_coordinates.get(x, [0, 0])[1])
        state_data = state_data[(state_data["lat"] != 0) & (state_data["lon"] != 0)]
    
        # Dropdown for map visualization
        map_option = st.selectbox("Choose map visualization:", ["Total Sales", "Top Category"])
        if map_option == "Total Sales":
            layer = pdk.Layer(
                "ColumnLayer",
                data=state_data,
                get_position=["lon", "lat"],
                get_elevation="total_sales",
                elevation_scale=5,
                radius=50000,
                get_fill_color="[200, 30, 0, 160]",
                pickable=True,
                auto_highlight=True,
            )
        else:
            state_data["category_numeric"] = pd.Categorical(state_data["top_category"]).codes
            layer = pdk.Layer(
                "ColumnLayer",
                data=state_data,
                get_position=["lon", "lat"],
                get_elevation="category_numeric",
                elevation_scale=50,
                radius=50000,
                get_fill_color="[category_numeric * 50, 100, 200]",
                pickable=True,
                auto_highlight=True,
            )
    
        # View configuration
        view_state = pdk.ViewState(latitude=37.5, longitude=-98.35, zoom=3, pitch=45)
    
        # Deck visualization with tooltip
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "html": """
                    <b>State:</b> {State_x} <br>
                    <b>Total Sales:</b> {total_sales} <br>
                    <b>Total Buyers:</b> {total_buyers} <br>
                    <b>Top Category:</b> {top_category}
                """,
                "style": {"backgroundColor": "steelblue", "color": "white"},
            },
        )
        st.pydeck_chart(r)


    # EDA Section
    if visualization == "EDA":
        st.write("### Exploratory Data Analysis (EDA)")
        tabs = st.tabs(
            [
                "Price Analysis",
                "Sales and Revenue Trends",
                "Order Trends",
                "Category Analysis",
                "Correlation Analysis",
                "Seasonal Trends",
                "Ratings Analysis",
            ]
        )

        # Tab 1: Price Analysis
        with tabs[0]:
            st.subheader("Price Analysis")

            # Distribution of Prices (Without Outliers)
            st.write("### Distribution of Prices (Without Outliers)")
            upper_limit = data_cleaned['price'].quantile(0.95)
            filtered_data = data_cleaned[data_cleaned['price'] <= upper_limit]
            fig_dist = px.histogram(
                filtered_data,
                x='price',
                nbins=30,
                title="Distribution of Prices (Without Outliers)",
                labels={'price': 'Price'},
            )
            st.plotly_chart(fig_dist)

            # Boxplot of Prices (Without Outliers)
            st.write("### Boxplot of Prices (Without Outliers)")
            fig_box = px.box(
                filtered_data,
                y='price',
                title="Boxplot of Prices (Without Outliers)",
                labels={'price': 'Price'},
            )
            st.plotly_chart(fig_box)

            # Average Price by Region
            st.write("### Average Price by Region")
            avg_price_region = data_cleaned.groupby('Region')['price'].mean().reset_index()
            
            # Add a toggle for chart type
            chart_type = st.radio(
                "Select chart type for Average Price by Region:",
                options=["Bar Chart", "Line Graph"]
            )
            
            # Conditional Chart Rendering
            if chart_type == "Bar Chart":
                fig_avg_price = px.bar(
                    avg_price_region,
                    x='Region',
                    y='price',
                    title="Average Price by Region (Bar Chart)",
                    labels={'price': 'Average Price', 'Region': 'Region'},
                )
            else:  # Line Graph
                fig_avg_price = px.line(
                    avg_price_region,
                    x='Region',
                    y='price',
                    title="Average Price by Region (Line Graph)",
                    labels={'price': 'Average Price', 'Region': 'Region'},
                    markers=True
                )
            
            # Display the chart
            st.plotly_chart(fig_avg_price)


        # Tab 2: Sales and Revenue Trends
        with tabs[1]:
            st.subheader("Sales and Revenue Trends")

            # Orders Over Time (Monthly)
            st.write("### Orders Over Time (Monthly)")
            monthly_orders = data_cleaned.groupby('year_month').size().reset_index(name='Order Count')
            
            # Add a toggle for chart type
            chart_type = st.radio(
                "Select chart type for Orders Over Time:",
                options=["Line Graph", "Bar Chart"]
            )
            
            # Conditional Chart Rendering
            if chart_type == "Line Graph":
                fig_orders = px.line(
                    monthly_orders,
                    x='year_month',
                    y='Order Count',
                    title="Orders Over Time (Line Graph)",
                    labels={'year_month': 'Year-Month', 'Order Count': 'Order Count'},
                    markers=True
                )
            else:  # Bar Chart
                fig_orders = px.bar(
                    monthly_orders,
                    x='year_month',
                    y='Order Count',
                    title="Orders Over Time (Bar Chart)",
                    labels={'year_month': 'Year-Month', 'Order Count': 'Order Count'},
                )
            
            # Display the chart
            st.plotly_chart(fig_orders)


            # Total Sales by Month
            st.write("### Total Sales by Month")
            total_sales_by_month = data_cleaned.groupby('month')['total'].sum().reset_index()
            
            # Add a toggle for chart type
            chart_type = st.radio(
                "Select chart type for Total Sales by Month:",
                options=["Bar Chart", "Line Graph"]
            )
            
            # Conditional Chart Rendering
            if chart_type == "Bar Chart":
                fig_sales = px.bar(
                    total_sales_by_month,
                    x='month',
                    y='total',
                    title="Total Sales by Month (Bar Chart)",
                    labels={'month': 'Month', 'total': 'Total Sales'},
                )
            else:  # Line Graph
                fig_sales = px.line(
                    total_sales_by_month,
                    x='month',
                    y='total',
                    title="Total Sales by Month (Line Graph)",
                    labels={'month': 'Month', 'total': 'Total Sales'},
                    markers=True
                )
            
            # Display the chart
            st.plotly_chart(fig_sales)


            # Total Sales by All Categories
            st.write("### Total Sales by All Categories")
            category_sales = data_cleaned.groupby('category')['total'].sum().sort_values(ascending=False).reset_index()
            fig_total_sales = px.bar(
                category_sales,
                x='total',
                y='category',
                title="Total Sales for All Categories",
                orientation='h',
                labels={'category': 'Category', 'total': 'Total Sales'},
            )
            st.plotly_chart(fig_total_sales)

        # Tab 3: Order Trends
        with tabs[2]:
            st.subheader("Order Trends")

            # Count of Orders per Region
            st.write("### Count of Orders per Region")
            orders_per_region = data_cleaned['Region'].value_counts().reset_index()
            orders_per_region.columns = ['Region', 'Order Count']
            
            # Add a toggle for chart type
            chart_type = st.radio(
                "Select chart type for Count of Orders per Region:",
                options=["Bar Chart", "Line Graph"]
            )
            
            # Conditional Chart Rendering
            if chart_type == "Bar Chart":
                fig_orders_region = px.bar(
                    orders_per_region,
                    x='Region',
                    y='Order Count',
                    title="Count of Orders per Region (Bar Chart)",
                    labels={'Order Count': 'Order Count'},
                )
            else:  # Line Graph
                fig_orders_region = px.line(
                    orders_per_region,
                    x='Region',
                    y='Order Count',
                    title="Count of Orders per Region (Line Graph)",
                    labels={'Order Count': 'Order Count'},
                    markers=True
                )
            
            # Display the chart
            st.plotly_chart(fig_orders_region)


            # Number of Orders by Month (Seasonal Trend)
            st.write("### Number of Orders by Month (Seasonal Trend)")
            orders_by_month = data_cleaned.groupby('month').size().reset_index(name='Order Count')
            
            # Add a toggle for chart type
            chart_type = st.radio(
                "Select chart type for Number of Orders by Month:",
                options=["Line Graph", "Bar Chart"]
            )
            
            # Conditional Chart Rendering
            if chart_type == "Line Graph":
                fig_orders_month = px.line(
                    orders_by_month,
                    x='month',
                    y='Order Count',
                    title="Number of Orders by Month (Line Graph)",
                    labels={'month': 'Month', 'Order Count': 'Order Count'},
                    markers=True
                )
            else:  # Bar Chart
                fig_orders_month = px.bar(
                    orders_by_month,
                    x='month',
                    y='Order Count',
                    title="Number of Orders by Month (Bar Chart)",
                    labels={'month': 'Month', 'Order Count': 'Order Count'},
                )
            
            # Display the chart
            st.plotly_chart(fig_orders_month)


        # Tab 4: Category Analysis
        with tabs[3]:
            st.subheader("Category Analysis")

            # Total Sales by Category
            st.write("### Total Sales by Category")
            category_sales = data_cleaned.groupby('category')['total'].sum().reset_index()
            
            # Add a toggle for chart type
            chart_type = st.radio(
                "Select chart type for Total Sales by Category:",
                options=["Bar Chart", "Line Graph"]
            )
            
            # Conditional Chart Rendering
            if chart_type == "Bar Chart":
                fig_category_sales = px.bar(
                    category_sales,
                    x='category',
                    y='total',
                    title="Total Sales by Category (Bar Chart)",
                    labels={'category': 'Category', 'total': 'Total Sales'},
                )
            else:  # Line Graph
                fig_category_sales = px.line(
                    category_sales,
                    x='category',
                    y='total',
                    title="Total Sales by Category (Line Graph)",
                    labels={'category': 'Category', 'total': 'Total Sales'},
                    markers=True
                )
            
            # Display the chart
            st.plotly_chart(fig_category_sales)


            # Top 5 Best Performing Categories
            st.write("### Top 5 Best Performing Categories")
            top_5_categories = category_sales.nlargest(5, 'total')
            
            # Add a toggle for chart type
            chart_type = st.radio(
                "Select chart type for Top 5 Best Performing Categories:",
                options=["Bar Chart", "Line Graph"]
            )
            
            # Conditional Chart Rendering
            if chart_type == "Bar Chart":
                fig_top_5 = px.bar(
                    top_5_categories,
                    x='category',
                    y='total',
                    title="Top 5 Categories by Total Sales (Bar Chart)",
                    labels={'category': 'Category', 'total': 'Total Sales'},
                )
            else:  # Line Graph
                fig_top_5 = px.line(
                    top_5_categories,
                    x='category',
                    y='total',
                    title="Top 5 Categories by Total Sales (Line Graph)",
                    labels={'category': 'Category', 'total': 'Total Sales'},
                    markers=True
                )
            
            # Display the chart
            st.plotly_chart(fig_top_5)


            # Filter by Product Category
            st.write("### Filter by Product Category")
            category_options = data_cleaned['category'].dropna().unique()
            selected_category = st.selectbox("Select Category", category_options)
            
            category_filtered_data = data_cleaned[data_cleaned['category'] == selected_category]
            filtered_sales = category_filtered_data.groupby('month')['total'].sum().reset_index()
            
            # Add a toggle for chart type
            chart_type = st.radio(
                f"Select chart type for Total Sales of {selected_category} by Month:",
                options=["Bar Chart", "Line Graph"]
            )
            
            # Conditional Chart Rendering
            if chart_type == "Bar Chart":
                fig_filtered_sales = px.bar(
                    filtered_sales,
                    x='month',
                    y='total',
                    title=f"Total Sales for {selected_category} by Month (Bar Chart)",
                    labels={'month': 'Month', 'total': 'Total Sales'},
                )
            else:  # Line Graph
                fig_filtered_sales = px.line(
                    filtered_sales,
                    x='month',
                    y='total',
                    title=f"Total Sales for {selected_category} by Month (Line Graph)",
                    labels={'month': 'Month', 'total': 'Total Sales'},
                    markers=True
                )
            
            # Display the chart
            st.plotly_chart(fig_filtered_sales)


        # Tab 5: Correlation Analysis
        with tabs[4]:
            st.subheader("Correlation Analysis")

            # Correlation Heatmap
            st.write("### Correlation Heatmap")
            numeric_cols = data_cleaned.select_dtypes(include=[np.number])
            correlation_matrix = numeric_cols.corr()
            fig_correlation = px.imshow(
                correlation_matrix,
                text_auto=True,
                color_continuous_scale="RdBu",
                title="Correlation Heatmap",
            )
            st.plotly_chart(fig_correlation)

        # Tab 6: Seasonal Trends
        with tabs[5]:
            st.subheader("Seasonal Trends")

            # Average Price by Month
            st.write("### Average Price by Month")
            avg_price_by_month = data_cleaned.groupby('month')['price'].mean().reset_index()
            
            # Add a toggle for chart type
            chart_type = st.radio(
                "Select chart type for Average Price by Month:",
                options=["Line Graph", "Bar Chart"]
            )
            
            # Conditional Chart Rendering
            if chart_type == "Line Graph":
                fig_avg_price_month = px.line(
                    avg_price_by_month,
                    x='month',
                    y='price',
                    title="Average Price by Month (Line Graph)",
                    labels={'month': 'Month', 'price': 'Average Price'},
                    markers=True
                )
            else:  # Bar Chart
                fig_avg_price_month = px.bar(
                    avg_price_by_month,
                    x='month',
                    y='price',
                    title="Average Price by Month (Bar Chart)",
                    labels={'month': 'Month', 'price': 'Average Price'},
                )
            
            # Display the chart
            st.plotly_chart(fig_avg_price_month)


            # Total Sales by Month
            st.write("### Total Sales by Month")
            total_sales_by_month = data_cleaned.groupby('month')['total'].sum().reset_index()
            
            # Add a toggle for chart type
            chart_type = st.radio(
                "Select chart type for Total Sales by Month:",
                options=["Line Graph", "Bar Chart"]
            )
            
            # Conditional Chart Rendering
            if chart_type == "Line Graph":
                fig_sales_by_month = px.line(
                    total_sales_by_month,
                    x='month',
                    y='total',
                    title="Total Sales by Month (Line Graph)",
                    labels={'month': 'Month', 'total': 'Total Sales'},
                    markers=True
                )
            else:  # Bar Chart
                fig_sales_by_month = px.bar(
                    total_sales_by_month,
                    x='month',
                    y='total',
                    title="Total Sales by Month (Bar Chart)",
                    labels={'month': 'Month', 'total': 'Total Sales'},
                )
            
            # Display the chart
            st.plotly_chart(fig_sales_by_month)


            # Seasonal Trends by Category
            st.write("### Seasonal Trends by Category")
            category_trends = data_cleaned.groupby(['month', 'category'])['total'].sum().reset_index()
            fig_category_trends = px.line(
                category_trends,
                x='month',
                y='total',
                color='category',
                title="Seasonal Trends by Category",
                labels={'month': 'Month', 'total': 'Total Sales', 'category': 'Category'},
            )
            st.plotly_chart(fig_category_trends)

        # Tab 7: Ratings Analysis
        with tabs[6]:
            st.subheader("Ratings Analysis")

            # Distribution of Ratings
            st.write("### Distribution of Ratings")
            stars_df = raw_data[['stars']].dropna()
            fig_ratings = px.histogram(
                stars_df,
                x='stars',
                nbins=30,
                title="Distribution of Ratings",
                labels={'stars': 'Stars (Ratings)'},
            )
            st.plotly_chart(fig_ratings)

    # Model Development Section
    if visualization == "Model Development":
        st.write("### Model Development")
        model_choice = st.selectbox(
            "Select a Model for Development:",
            [
                "Gradient Boosting Insights",
                "Random Forest Insights",
                "Linear Regression Insights",
                "K-Means Clustering Insights",
            ],
        )

        # Verify column existence and handle missing columns
        def validate_columns(columns):
            missing_columns = [col for col in columns if col not in data.columns]
            if missing_columns:
                st.error(f"Missing columns in the dataset: {missing_columns}")
                return False
            return True

        if model_choice == "Gradient Boosting Insights":
            st.write("### Gradient Boosting Regressor Insights")
            feature_columns = ['age', 'qty_ordered', 'discount_amount', 'price_x', 'category_encoded']
            if validate_columns(feature_columns + ['total']):
                X = data[feature_columns].dropna()
                y = data['total'].dropna()
        
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                gbr = GradientBoostingRegressor(random_state=42)
                gbr.fit(X_train, y_train)
        
                y_pred = gbr.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                adjusted_r2 = 1 - (1 - r2) * ((X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1))
        
                st.write(f"**R²:** {r2:.2f}, **Adjusted R²:** {adjusted_r2:.2f}, **RMSE:** {np.sqrt(mse):.2f}")
        
                # Feature Importance Visualization with Toggle
                chart_type = st.radio(
                    "Select chart type for Feature Importances:",
                    options=["Bar Chart", "Line Graph"]
                )
        
                feature_importances = gbr.feature_importances_
                if chart_type == "Bar Chart":
                    importance_fig = px.bar(
                        x=feature_columns,
                        y=feature_importances,
                        labels={'x': 'Feature', 'y': 'Importance'},
                        title="Gradient Boosting Feature Importances",
                    )
                elif chart_type == "Line Graph":
                    importance_fig = px.line(
                        x=feature_columns,
                        y=feature_importances,
                        labels={'x': 'Feature', 'y': 'Importance'},
                        title="Gradient Boosting Feature Importances",
                        markers=True
                    )
        
                st.plotly_chart(importance_fig)
        
                # Residual Histogram
                residuals = y_test - y_pred
                residuals_fig = px.histogram(residuals, nbins=30, title="Residual Histogram")
                st.plotly_chart(residuals_fig)



                # Predicted vs Actual
                pred_actual_fig = px.scatter(
                    x=y_test,
                    y=y_pred,
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    title="Predicted vs Actual (Gradient Boosting)",
                )

                # Add a dotted line for perfect prediction (where y_test = y_pred)
                pred_actual_fig.add_shape(
                type="line",
                x0=min(y_test),  # Starting point on x-axis (min of actual values)
                y0=min(y_test),  # Starting point on y-axis (min of actual values)
                x1=max(y_test),  # Ending point on x-axis (max of actual values)
                y1=max(y_test),  # Ending point on y-axis (max of actual values)
                line=dict(
                    color="red",  # Color of the line
                    dash="dot",  # Dotted line style
                    width=2,  # Line width
                    ),
                )
                st.plotly_chart(pred_actual_fig)

        elif model_choice == "Random Forest Insights":
            st.write("### Random Forest Regressor Insights")
            feature_columns = ['price_x', 'discount_amount', 'category_encoded', 'Region_encoded_x']
            if validate_columns(feature_columns + ['qty_ordered']):
                X = data[feature_columns].fillna(data[feature_columns].median())
                y = data['qty_ordered']
        
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
        
                y_pred = rf_model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                adjusted_r2 = 1 - (1 - r2) * ((X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1))
        
                st.write(f"**R²:** {r2:.2f}, **Adjusted R²:** {adjusted_r2:.2f}")
        
                # Add a toggle for Feature Importances
                chart_type = st.radio(
                    "Select chart type for Feature Importances:",
                    options=["Bar Chart", "Line Graph"]
                )
        
                # Feature Importances Visualization
                feature_importances = rf_model.feature_importances_
                if chart_type == "Bar Chart":
                    fig = px.bar(
                        x=feature_columns,
                        y=feature_importances,
                        labels={'x': 'Feature', 'y': 'Importance'},
                        title="Random Forest Feature Importances (Bar Chart)",
                    )
                else:  # Line Graph
                    fig = px.line(
                        x=feature_columns,
                        y=feature_importances,
                        labels={'x': 'Feature', 'y': 'Importance'},
                        title="Random Forest Feature Importances (Line Graph)",
                        markers=True
                    )
        
                st.plotly_chart(fig)


                # Residual Histogram
                residuals = y_test - y_pred
                fig = px.histogram(residuals, nbins=30, title="Residual Histogram")
                st.plotly_chart(fig)

                # Predicted vs Actual
                fig = px.scatter(
                    x=y_test,
                    y=y_pred,
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    title="Predicted vs Actual (Random Forest)",
                )

                # Add a dotted line for perfect prediction (where y_test = y_pred)
                fig.add_shape(
                    type="line",
                    x0=min(y_test),  # Starting point on x-axis (min of actual values)
                    y0=min(y_test),  # Starting point on y-axis (min of actual values)
                    x1=max(y_test),  # Ending point on x-axis (max of actual values)
                    y1=max(y_test),  # Ending point on y-axis (max of actual values)
                    line=dict(
                        color="red",  # Color of the line
                        dash="dot",  # Dotted line style
                        width=2,  # Line width
                    ),
                )                

                
                st.plotly_chart(fig)

        elif model_choice == "Linear Regression Insights":
            st.write("### Linear Regression Insights")
            feature_columns = ['price_x', 'discount_amount', 'category_encoded', 'Region_encoded_x']
            if validate_columns(feature_columns + ['qty_ordered']):
                X = data[feature_columns].fillna(0)
                y = data['qty_ordered']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                lr_model = LinearRegression()
                lr_model.fit(X_train, y_train)

                y_pred = lr_model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)

                st.write(f"**R²:** {r2:.2f}, **RMSE:** {np.sqrt(mse):.2f}")

                # Residual Histogram
                residuals = y_test - y_pred
                fig = px.histogram(residuals, nbins=30, title="Residual Histogram (Linear Regression)")
                st.plotly_chart(fig)

                # Predicted vs Actual
                fig = px.scatter(
                    x=y_test,
                    y=y_pred,
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    title="Predicted vs Actual (Linear Regression)",
                )

                # Add a dotted line for perfect prediction (where y_test = y_pred)
                fig.add_shape(
                    type="line",
                    x0=min(y_test),  # Starting point on x-axis (min of actual values)
                    y0=min(y_test),  # Starting point on y-axis (min of actual values)
                    x1=max(y_test),  # Ending point on x-axis (max of actual values)
                    y1=max(y_test),  # Ending point on y-axis (max of actual values)
                    line=dict(
                        color="red",  # Color of the line
                        dash="dot",  # Dotted line style
                        width=2,  # Line width
                    ),
                )

                st.plotly_chart(fig)

        elif model_choice == "K-Means Clustering Insights":
            st.write("### K-Means Clustering Insights")
            feature_columns = ['price_x', 'qty_ordered']
            if validate_columns(feature_columns):
                clustering_data = data[feature_columns].dropna()

                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(clustering_data)

                # Elbow Method for Optimal Clusters
                st.write("#### Elbow Method for Optimal Clusters")
                numerical_columns = [
                    "qty_ordered",
                    "price_x",
                    "value",
                    "total",
                    "category_encoded",
                    "discount_amount",
                    "Region_encoded_x",
                    "payment_method_encoded",
                ]
                
                # Validate feature columns exist in the dataset
                if validate_columns(numerical_columns):
                    clustering_data = data[numerical_columns].dropna()
                
                    # Standardize the data
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(clustering_data)
                
                    # Calculate inertia for different cluster sizes
                    inertia = []
                    range_clusters = range(1, 11)
                    for k in range_clusters:
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        kmeans.fit(scaled_data)
                        inertia.append(kmeans.inertia_)
                
                    # Interactive Elbow Plot
                    elbow_fig = px.line(
                        x=list(range_clusters),
                        y=inertia,
                        labels={"x": "Number of Clusters (k)", "y": "Inertia"},
                        title="Elbow Method for Optimal Clusters"
                    )
                    elbow_fig.update_traces(mode="lines+markers")
                
                    # Display in Streamlit
                    st.plotly_chart(elbow_fig)
                else:
                    st.error("Some of the required columns for clustering are missing in the dataset.")

                # Perform K-Means Clustering
                num_clusters = 4  # Replace with your desired number of clusters
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                clustering_data['Cluster'] = kmeans.fit_predict(scaled_data)
                
                # Clusters based on Price and Quantity Ordered
                st.write("#### Clusters Based on Price and Quantity Ordered")
                cluster_scatter = px.scatter(
                    clustering_data,
                    x='price_x',
                    y='qty_ordered',
                    color='Cluster',
                    title="Clusters: Price vs Quantity Ordered",
                    labels={'price_x': 'Price', 'qty_ordered': 'Quantity Ordered', 'Cluster': 'Cluster'},
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                )
                st.plotly_chart(cluster_scatter)


                import plotly.express as px
                import pandas as pd
                import streamlit as st
                
                # Assuming 'clustering_data' is already defined and includes the necessary columns
                
                # Grouped bar chart for average values per category per cluster
                category_cluster_summary = clustering_data.groupby(['category_encoded', 'Cluster']).mean().reset_index()
                
                # Create an interactive grouped bar chart using Plotly
                bar_chart = px.bar(
                    category_cluster_summary,
                    x='category_encoded',
                    y='price_x',
                    color='Cluster',
                    title='Average Price per Category Across Clusters',
                    labels={'category_encoded': 'Category (Encoded)', 'price_x': 'Average Price', 'Cluster': 'Cluster'},
                    barmode='group',
                    color_discrete_sequence=px.colors.qualitative.Set1  # Set1 color palette for better visuals
                )
                
                # Display the interactive plot in Streamlit
                st.plotly_chart(bar_chart)