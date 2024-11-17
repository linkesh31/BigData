import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# App Title
st.title("Project Dashboard")
st.markdown("This dashboard presents detailed insights on the dataset.")

# Authentication
USERNAME = "group6"
PASSWORD = "group6"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.sidebar.success("Login successful!")
        else:
            st.sidebar.error("Invalid username or password.")

if st.session_state.logged_in:
    st.sidebar.success("Logged in as group6")
    dataset_path = "data/preprocessed_combined_dataset.csv"

    @st.cache_data
    def load_data():
        return pd.read_csv(dataset_path)

    try:
        data = load_data()

        # Dropdown Menu for Insights
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
                "Model Development",
                "EDA",
            ],
        )

        if visualization == "Dataset Overview":
            st.write("### Dataset Overview")
            st.dataframe(data.head())

        elif visualization == "Best Performing Category":
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

        elif visualization == "Purchase Difference Between Years":
            st.write("### Purchase Difference Between Years")
            year_comparison = data.groupby("year")["total"].sum().reset_index()
            st.line_chart(year_comparison.set_index("year"))

        elif visualization == "Purchase Analysis by Gender":
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

        elif visualization == "Purchase Analysis by Age Group":
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

        elif visualization == "3D Map of Purchases by State":
            st.write("### 3D Map of Purchases by State")
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
            state_data = data.groupby("State_x").agg(
                total_sales=("total", "sum"),
                total_buyers=("cust_id", "nunique"),
                top_category=("category", lambda x: x.mode()[0]),
            ).reset_index()

            state_data["lat"] = state_data["State_x"].map(lambda x: state_coordinates.get(x, [0, 0])[0])
            state_data["lon"] = state_data["State_x"].map(lambda x: state_coordinates.get(x, [0, 0])[1])
            state_data = state_data[(state_data["lat"] != 0) & (state_data["lon"] != 0)]

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
                )

            view_state = pdk.ViewState(latitude=37.5, longitude=-98.35, zoom=3, pitch=45)
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    "html": "<b>State:</b> {State_x} <br><b>Total Sales:</b> {total_sales}<br><b>Top Category:</b> {top_category}",
                },
            )
            st.pydeck_chart(r)

        elif visualization == "Model Development":
            st.write("### Model Development")
            model_choice = st.selectbox(
                "Select a Model for Development:",
                [
                    "Random Forest Regressor",
                    "Regression",
                    "Gradient Boosting Regressor",
                    "K-Means Clustering",
                ],
            )

            if model_choice == "Random Forest Regressor":
                st.write("#### Random Forest Regressor")
                feature_importances = {"Feature": ["Age", "Gender", "Category", "State"], "Importance": [0.4, 0.3, 0.2, 0.1]}
                feature_df = pd.DataFrame(feature_importances)
                fig = px.bar(feature_df, x="Feature", y="Importance", title="Feature Importance in Random Forest")
                st.plotly_chart(fig)

            elif model_choice == "Regression":
                st.write("#### Regression")
                simulated_data = pd.DataFrame(
                    {"Actual": [10, 15, 20, 25, 30], "Predicted": [12, 14, 19, 26, 29]}
                )
                fig = px.scatter(
                    simulated_data,
                    x="Actual",
                    y="Predicted",
                    title="Regression Model: Actual vs Predicted",
                )
                st.plotly_chart(fig)

            elif model_choice == "Gradient Boosting Regressor":
                st.write("#### Gradient Boosting Regressor")
                simulated_data = pd.DataFrame(
                    {"Actual": [10, 15, 20, 25, 30], "Predicted": [11, 16, 19, 24, 31]}
                )
                fig = px.scatter(
                    simulated_data,
                    x="Actual",
                    y="Predicted",
                    title="Gradient Boosting Regressor: Actual vs Predicted",
                )
                st.plotly_chart(fig)

            elif model_choice == "K-Means Clustering":
                st.write("#### K-Means Clustering")
                cluster_data = pd.DataFrame(
                    {"x": [1, 2, 3, 8, 9, 10], "y": [1, 2, 3, 8, 9, 10], "Cluster": [1, 1, 1, 2, 2, 2]}
                )
                fig = px.scatter(cluster_data, x="x", y="y", color="Cluster")
                st.plotly_chart(fig)

        elif visualization == "EDA":
            st.write("### EDA")
            eda_choice = st.selectbox(
                "Select EDA Visualization:",
                [
                    "Count of Orders by Region",
                    "Price vs Discount",
                    "Correlation Heatmap",
                    "Orders Over Time",
                ],
            )

            if eda_choice == "Count of Orders by Region":
                if "Region_x" in data.columns:
                    plt.figure(figsize=(10, 6))
                    sns.countplot(y=data["Region_x"], order=data["Region_x"].value_counts().index)
                    plt.title("Count of Orders by Region")
                    st.pyplot(plt)

            elif eda_choice == "Price vs Discount":
                if "price_x" in data.columns and "discount_amount" in data.columns:
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x="price_x", y="discount_amount", data=data)
                    plt.title("Price vs Discount")
                    st.pyplot(plt)

            elif eda_choice == "Correlation Heatmap":
                numeric_cols = data.select_dtypes(include=["float", "int"])
                # Filter out low variance columns
                numeric_cols = numeric_cols.loc[:, numeric_cols.var() > 0.1]

                if not numeric_cols.empty:
                    plt.figure(figsize=(12, 8))
                    heatmap = sns.heatmap(
                        numeric_cols.corr(),
                        annot=True,
                        fmt=".2f",
                        cmap="coolwarm",
                        cbar=True,
                        linewidths=0.5,
                    )
                    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
                    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
                    plt.title("Correlation Heatmap")
                    st.pyplot(plt)
                else:
                    st.error("No numeric columns with sufficient variance available for correlation analysis.")

            elif eda_choice == "Orders Over Time":
                if "order_date" in data.columns:
                    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
                    data["month_year"] = data["order_date"].dt.to_period("M")
                    orders_over_time = data.groupby("month_year").size()
                    plt.figure(figsize=(10, 6))
                    orders_over_time.plot(kind="line", marker="o")
                    plt.title("Orders Over Time")
                    plt.xlabel("Month-Year")
                    plt.ylabel("Order Count")
                    st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred: {e}")