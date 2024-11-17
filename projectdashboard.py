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

            # Displaying the diagrams after dataset overview
            st.write("### Project Diagrams")
            
            st.subheader("Context Diagram")
            st.image("appendix/DFD.png", caption="Context Diagram", use_column_width=True)

            st.subheader("Flowchart")
            st.image("appendix/Flowchart.png", caption="Flowchart", use_column_width=True)

            st.subheader("Entity-Relationship Diagram (ERD)")
            st.image("appendix/ERD.png", caption="ERD", use_column_width=True)

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

        # Additional visualizations and model development sections go here...

    except Exception as e:
        st.error(f"An error occurred: {e}")
