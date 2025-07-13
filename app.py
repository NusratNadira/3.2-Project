import streamlit as st
import pandas as pd
import numpy as np
from models.model_code import preprocess_data, create_logistic_regression_model, train_logistic_regression_model
from models.federated_learning import FederatedLearning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from models.model_code import preprocess_data, create_logistic_regression_model, train_logistic_regression_model
from models.federated_learning import FederatedLearning

# Streamlit UI
st.title("Federated Learning with Logistic Regression")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Loaded", df.head())

    # Preprocess the dataset
    df_final = preprocess_data(df)
    st.write("Preprocessed Data", df_final.head())

    # Split the dataset into features (X) and labels (y)
    y = df_final['label']
    X = df_final.drop('label', axis=1)
    
    # Label Encoding and One-Hot Encoding
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    num_classes = len(np.unique(y))
    y = to_categorical(y)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)
    st.write(f"Training Data Shape: {X_train.shape}")
    st.write(f"Test Data Shape: {X_test.shape}")

    # Choose model for training
    if st.button("Train Logistic Regression Model"):
        # Create and train logistic regression model
        model = create_logistic_regression_model(X_train.shape[1], num_classes)
        accuracy = train_logistic_regression_model(X_train, y_train, X_test, y_test, model)
        st.write(f"Model Accuracy: {accuracy:.4f}")

    # Federated Learning Button (implement training)
    if st.button("Start Federated Learning"):
        # Initialize Federated Learning with 4 clients
        fl = FederatedLearning(num_clients=4, input_dim=X_train.shape[1], num_classes=num_classes, client_epochs=5, rounds=5)
        
        # Start Federated Learning
        global_model = fl.train(X_train, y_train, X_test, y_test)
        st.write("Federated Learning Completed")
        st.write("Global Model Accuracy:", fl.evaluate_global_model(X_test, y_test))
