from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

# Preprocessing data (already defined in `app.py`)
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
