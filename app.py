import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# --- 1. DATA GENERATION ---
# This part of the script will generate a sample CSV file if it doesn't exist.
# This makes the app self-contained and easy to run.

def generate_data(filename="student_performance.csv"):
    """Generates a synthetic student performance dataset."""
    if not os.path.exists(filename):
        print(f"Generating synthetic data: {filename}")
        num_records = 200
        np.random.seed(42) # for reproducibility
        
        # Feature Generation
        study_hours = np.random.uniform(1, 20, num_records)
        previous_grades = np.random.uniform(40, 100, num_records)
        attendance_percentage = np.random.uniform(70, 100, num_records)
        extracurricular_activities = np.random.randint(0, 2, num_records) # 0 for No, 1 for Yes
        
        # Target variable (Final Score) with some noise
        # The final score is logically dependent on the features.
        noise = np.random.normal(0, 5, num_records)
        final_score = (
            25  # Base score
            + (study_hours * 2.5)
            + (previous_grades * 0.4)
            + (attendance_percentage * 0.15)
            + (extracurricular_activities * 1.5)
            + noise
        )
        
        # Ensure scores are within the 0-100 range
        final_score = np.clip(final_score, 0, 100)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Study_Hours': study_hours,
            'Previous_Grades': previous_grades,
            'Attendance_Percentage': attendance_percentage,
            'Extracurricular_Activities': extracurricular_activities,
            'Final_Score': final_score
        })
        
        df.to_csv(filename, index=False)
        print("Data generation complete.")
    else:
        print(f"Data file already exists: {filename}")

# Generate the data file
generate_data()

# --- 2. STREAMLIT APP ---

# --- App Configuration ---
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# --- Load Data ---
@st.cache_data
def load_data(filepath="student_performance.csv"):
    """Loads the student performance data."""
    return pd.read_csv(filepath)

df = load_data()

# --- Title and Introduction ---
st.title("🎓 Student Performance Prediction")
st.write("""
This application analyzes student data and predicts their final exam scores based on several factors. 
Use the sidebar to input student details and get a prediction. The sections below explore the underlying data.
""")


# --- 3. PREDICTIVE MODEL ---

# --- Sidebar for User Input ---
st.sidebar.header("🔮 Predict Final Score")
st.sidebar.write("Adjust the sliders and options to see the predicted final score for a student.")

# Input fields in the sidebar
input_study_hours = st.sidebar.slider("Study Hours per Week", 1.0, 20.0, 10.0, 0.5)
input_previous_grades = st.sidebar.slider("Previous Semester Grade (%)", 40.0, 100.0, 80.0, 1.0)
input_attendance = st.sidebar.slider("Class Attendance (%)", 70.0, 100.0, 90.0, 1.0)
input_extracurricular = st.sidebar.selectbox("Involved in Extracurricular Activities?", ("No", "Yes"))

# Convert extracurricular input to numerical
input_extracurricular_numeric = 1 if input_extracurricular == "Yes" else 0

# --- Model Training ---
# Define features (X) and target (y)
features = ['Study_Hours', 'Previous_Grades', 'Attendance_Percentage', 'Extracurricular_Activities']
target = 'Final_Score'

X = df[features]
y = df[target]

# Split data (optional for this simple app, but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# --- Prediction Logic ---
# Create a DataFrame from user input
input_data = pd.DataFrame({
    'Study_Hours': [input_study_hours],
    'Previous_Grades': [input_previous_grades],
    'Attendance_Percentage': [input_attendance],
    'Extracurricular_Activities': [input_extracurricular_numeric]
})

# Get the prediction
prediction = model.predict(input_data)[0]
prediction = round(max(0, min(100, prediction)), 2) # Ensure prediction is within a valid score range

# Display the prediction prominently in the sidebar
st.sidebar.metric(
    label="Predicted Final Score",
    value=f"{prediction:.2f}%"
)

# Display model performance
y_pred_test = model.predict(X_test)
r2 = r2_score(y_test, y_pred_test)
st.sidebar.write(f"**Model R-squared:** `{r2:.2f}`")
st.sidebar.info("R-squared indicates the proportion of the variance in the final score that is predictable from the features. Higher is better.")


# --- 4. EXPLORATORY DATA ANALYSIS (EDA) ---
st.header("📊 Exploratory Data Analysis")
st.write("Let's explore the dataset to understand the relationships between different factors and student performance.")

# --- Data Display ---
with st.expander("View Raw Data"):
    st.dataframe(df)
with st.expander("View Data Summary"):
    st.write(df.describe())

# --- Visualizations ---
col1, col2 = st.columns(2)

with col1:
    # Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.write("""
    **Interpretation:** The heatmap shows the correlation between variables. 
    - **Positive values (red)** mean as one variable increases, the other tends to increase (e.g., `Study_Hours` and `Final_Score`).
    - **Negative values (blue)** indicate an inverse relationship.
    """)

with col2:
    # Feature Distributions
    st.subheader("Feature Distributions")
    feature_to_plot = st.selectbox("Select a feature to see its distribution:", df.columns)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df[feature_to_plot], kde=True, ax=ax)
    plt.title(f'Distribution of {feature_to_plot.replace("_", " ")}')
    st.pyplot(fig)
    st.write("""
    **Interpretation:** This histogram shows the frequency of different values for the selected feature, helping us understand its spread and central tendency.
    """)

# Relationship between features
st.subheader("Relationship Between Features and Final Score")
feature_for_scatter = st.selectbox(
    "Select a feature to plot against Final Score:",
    ['Study_Hours', 'Previous_Grades', 'Attendance_Percentage']
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(data=df, x=feature_for_scatter, y='Final_Score', ax=ax, line_kws={"color":"red"})
plt.title(f'{feature_for_scatter.replace("_", " ")} vs. Final Score')
st.pyplot(fig)
st.write(f"""
**Interpretation:** This scatter plot shows the relationship between `{feature_for_scatter.replace("_", " ")}` and the `Final_Score`. 
The red line is a regression line that indicates the general trend. A clear upward trend suggests a positive correlation.
""")

