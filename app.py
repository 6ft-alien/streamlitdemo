# -*- coding: utf-8 -*-
"""
Streamlit Web App for Diabetes Prediction and Visualization (using Seaborn)
"""

import numpy as np
import pandas as pd
import pickle
import os
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt

# --- MODEL TRAINING ---
# This part of the code remains the same. It trains and saves the model
# if it doesn't already exist. Using @st.cache_resource ensures this heavy
# computation runs only once.

MODEL_FILENAME = 'trained_model.sav'
DATASET_PATH = 'diabetes.csv'

@st.cache_resource
def train_and_save_model():
    """
    Trains an SVM classifier on the diabetes dataset and saves it to a file.
    If the model file already exists, it does nothing.
    """
    if not os.path.exists(MODEL_FILENAME):
        st.write("Model not found. Training a new one...")
        # Load data
        try:
            diabetes_dataset = pd.read_csv(DATASET_PATH)
        except FileNotFoundError:
            st.error(f"Error: The dataset file '{DATASET_PATH}' was not found.")
            st.stop()

        # Split data into features (X) and target (Y)
        X = diabetes_dataset.drop(columns='Outcome', axis=1)
        Y = diabetes_dataset['Outcome']
        X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

        # Train the Support Vector Machine (SVM) Classifier
        classifier = svm.SVC(kernel='linear', probability=True)
        classifier.fit(X_train, Y_train)

        # Save the trained model to a file
        with open(MODEL_FILENAME, 'wb') as f:
            pickle.dump(classifier, f)
        st.write("Model training complete and saved.")
    return MODEL_FILENAME

# Ensure the model is trained and available
train_and_save_model()

# --- DATA LOADING AND PREDICTION FUNCTION ---

@st.cache_data
def load_data():
    """Loads the diabetes dataset from CSV."""
    df = pd.read_csv(DATASET_PATH)
    return df

# Load the trained model
try:
    with open(MODEL_FILENAME, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
except FileNotFoundError:
    st.error(f"Model file '{MODEL_FILENAME}' not found. Please ensure the model is trained.")
    st.stop()


def diabetes_prediction(input_data):
    """
    Predicts diabetes using the loaded model.
    Returns the prediction text.
    """
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is **Not Diabetic**'
    else:
        return 'The person is **Diabetic**'


# --- STREAMLIT DASHBOARD UI ---

def main():
    st.set_page_config(layout="wide")
    
    # Load the dataset
    df = load_data()

    # --- SIDEBAR FOR PREDICTION ---
    with st.sidebar:
        st.header('Diabetes Prediction')
        st.write("Enter patient details to get a prediction.")

        # Input fields for prediction
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1, step=1)
        Glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=300, value=100)
        BloodPressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=150, value=70)
        SkinThickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
        Insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=900, value=80)
        BMI = st.number_input('BMI (kg/m²)', min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
        Age = st.number_input('Age (years)', min_value=0, max_value=120, value=30)

        # Prediction button and result
        if st.button('**Predict**', use_container_width=True):
            input_list = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            diagnosis = diabetes_prediction(input_list)
            st.success(diagnosis)

    # --- MAIN PAGE FOR VISUALIZATION ---
    st.title('Diabetes Analysis Dashboard')
    st.write("Explore the dataset and correlations between different health metrics.")

    # Display the raw data in an expandable section
    with st.expander("View Raw Data"):
        st.dataframe(df)

    # --- VISUALIZATIONS ---
    st.header("Data Visualizations")
    
    # Set plot style
    sns.set_style("whitegrid")

    # Layout with columns
    col1, col2 = st.columns(2)

    with col1:
        # Pie Chart for Outcome Distribution (using Matplotlib)
        st.subheader("Outcome Distribution")
        outcome_counts = df['Outcome'].value_counts()
        outcome_counts.index = ['Not Diabetic', 'Diabetic']
        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', startangle=90, colors=['#8FBC8F', '#F08080'])
        ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig_pie)

        # Histogram for a selected feature
        st.subheader("Feature Distributions")
        feature_to_plot = st.selectbox("Select a feature to view its distribution:", df.columns[:-1])
        fig_hist, ax_hist = plt.subplots()
        sns.histplot(data=df, x=feature_to_plot, hue='Outcome', kde=True, ax=ax_hist, palette=['#8FBC8F', '#F08080'])
        st.pyplot(fig_hist)

    with col2:
        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        corr = df.corr()
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax_heatmap)
        st.pyplot(fig_heatmap)

    # Scatter plot for exploring relationships
    st.header("Explore Relationships Between Features")
    col3, col4 = st.columns(2)
    with col3:
        x_axis = st.selectbox("Select X-axis feature:", df.columns[:-1], index=1) # Default to Glucose
    with col4:
        y_axis = st.selectbox("Select Y-axis feature:", df.columns[:-1], index=5) # Default to BMI

    fig_scatter, ax_scatter = plt.subplots()
    sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='Outcome', palette=['#8FBC8F', '#F08080'], ax=ax_scatter)
    st.pyplot(fig_scatter)


if __name__ == '__main__':
    main()
