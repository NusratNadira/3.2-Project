from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score

class FederatedLearning:
    def __init__(self, X_train, y_train, X_test, y_test, num_clients=4, epochs=5):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.num_clients = num_clients
        self.epochs = epochs
        self.global_model = LogisticRegression(max_iter=1000)  # Global model
        self.client_models = [LogisticRegression(max_iter=1000) for _ in range(num_clients)]  # Client models
        self.client_data = self.split_data_for_clients(X_train, y_train)

    def split_data_for_clients(self, X, y):
        # Split data across clients
        split_data = []
        num_samples = len(X) // self.num_clients
        for i in range(self.num_clients):
            start = i * num_samples
            end = (i + 1) * num_samples if i != self.num_clients - 1 else len(X)
            split_data.append((X[start:end], y[start:end]))
        return split_data

    def fedavg_aggregate(self):
        # Federated Averaging: Average weights of client models
        client_weights = [model.coef_ for model in self.client_models]
        averaged_weights = np.mean(client_weights, axis=0)
        self.global_model.coef_ = averaged_weights
        print("Global model updated using FedAvg.")

    def train(self):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            # Train each client model
            for i, (X_client, y_client) in enumerate(self.client_data):
                model = self.client_models[i]
                model.fit(X_client, y_client)
                y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                print(f"Client {i+1} Model Accuracy: {accuracy:.4f}")
            
            # Apply FedAvg aggregation
            self.fedavg_aggregate()
            
            # Evaluate the global model
            y_pred_global = self.global_model.predict(self.X_test)
            global_accuracy = accuracy_score(self.y_test, y_pred_global)
            print(f"Global Model Accuracy: {global_accuracy:.4f}")
            
            print("-" * 50)
        
        # After all epochs, return the global model
        return self.global_model
