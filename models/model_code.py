from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

# Preprocessing function
def preprocess_data(df):
    # Step 1: Remove labels with less than 100 records
    label_counts = df['label'].value_counts()
    labels_to_keep = label_counts[label_counts >= 100].index
    df_filtered = df[df['label'].isin(labels_to_keep)]

    # Step 2: Merge similar labels based on your mapping
    merge_mapping = {
        'DDoS': [
            'DDoS-ICMP_Flood', 'DDoS-UDP_Flood', 'DDoS-TCP_Flood',
            'DDoS-PSHACK_Flood', 'DDoS-RSTFINFlood', 'DDoS-SYN_Flood',
            'DDoS-SynonymousIP_Flood', 'DDoS-ICMP_Fragmentation',
            'DDoS-UDP_Fragmentation', 'DDoS-ACK_Fragmentation', 'DDoS-HTTP_Flood', 'DDoS-SlowLoris'
        ],
        'DoS': [
            'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-SYN_Flood', 'DoS-HTTP_Flood'
        ],
        'Mirai': [
            'Mirai-greeth_flood', 'Mirai-udpplain', 'Mirai-greip_flood'
        ],
        'Recon': [
            'Recon-HostDiscovery', 'Recon-OSScan', 'Recon-PortScan', 'Recon-PingSweep'
        ],
        'Other': [
            'MITM-ArpSpoofing', 'DNS_Spoofing', 'VulnerabilityScan', 'DictionaryBruteForce',
            'BrowserHijacking', 'CommandInjection', 'Backdoor_Malware', 'SqlInjection', 'XSS', 'Uploading_Attack'
        ]
    }

    reverse_mapping = {}
    for new_label, old_labels in merge_mapping.items():
        for old_label in old_labels:
            reverse_mapping[old_label] = new_label

    # Apply the mapping to create new labels
    df_filtered['label'] = df_filtered['label'].map(reverse_mapping).fillna(df_filtered['label'])

    # Step 3: Final dataset
    df_final = df_filtered

    return df_final

# Create Logistic Regression Model (for simplicity, it's a neural network here)
def create_logistic_regression_model(input_dim, num_classes):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_logistic_regression_model(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1, validation_data=(X_test, y_test))

    # Model evaluation
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred_classes == y_true)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Print classification report
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred_classes))

    return accuracy
