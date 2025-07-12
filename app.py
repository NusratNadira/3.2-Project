import streamlit as st
import pandas as pd
from federated_learning import FederatedLearning
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Streamlit UI: upload CSV file
def app():
    st.title("Federated Learning with Logistic Regression")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset loaded with shape:", df.shape)
        
        # Preprocess data
        X_train, X_test, y_train, y_test, label_classes = preprocess_data(df)
        
        # Initialize Federated Learning
        federated_learning = FederatedLearning(X_train, y_train, X_test, y_test)
        
        # Train the model using Federated Learning
        global_model = federated_learning.train()
        
        # SHAP analysis
        explain_with_shap(global_model, X_train, X_test)
        
        st.write("Global model training completed. SHAP analysis visualized above.")

# Preprocess data (same function as explained earlier)
def preprocess_data(df):
    attack_mapping = {
        'DDoS': ['DDoS-ICMP_Flood', 'DDoS-UDP_Flood', 'DDoS-TCP_Flood'],
        'DoS': ['DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-SYN_Flood'],
        'Mirai': ['Mirai-greeth_flood', 'Mirai-udpplain'],
        'Recon': ['Recon-HostDiscovery', 'Recon-PortScan'],
        'Other': ['MITM-ArpSpoofing', 'DNS_Spoofing', 'Backdoor_Malware'],
        'Benign': ['Normal']
    }

    reverse_mapping = {}
    for attack_type, attacks in attack_mapping.items():
        for attack in attacks:
            reverse_mapping[attack] = attack_type
    
    df['label'] = df['label'].map(reverse_mapping).fillna('Other')
    
    X = df.drop(columns=['label'])
    y = df['label']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, encoder.classes_

# SHAP Explanation (Same logic as in the notebook)
def explain_with_shap(model, X_train, X_test):
    explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])  # Sample for faster computation
    shap_values = explainer.shap_values(X_test[:100])  # First 100 test samples
    shap.summary_plot(shap_values, X_test[:100])
