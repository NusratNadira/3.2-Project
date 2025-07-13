import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout

class FederatedLearning:
    def __init__(self, num_clients=4, input_dim=None, num_classes=None, client_epochs=5, rounds=10):
        self.num_clients = num_clients  # Fixed to 4 clients
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.client_epochs = client_epochs
        self.rounds = rounds
        self.global_model = self.create_model(input_dim, num_classes)
        self.client_models = [self.create_model(input_dim, num_classes) for _ in range(num_clients)]

        print(f"Initialized Federated Learning with {self.num_clients} clients using FedAvg aggregation")

    def create_model(self, input_dim, num_classes):
        model = Sequential([
            Dense(128, input_dim=input_dim, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def fedavg_aggregate(self):
        """
        FedAvg (Federated Averaging) Aggregation Algorithm
        Computes weighted average of client model parameters
        """
        print("Applying FedAvg aggregation...")

        # Get weights from each client model
        client_weights = [model.get_weights() for model in self.client_models]

        # Initialize averaged weights structure
        averaged_weights = []

        # Iterate through each layer's weights
        for layer_idx in range(len(client_weights[0])):
            # Get the weights for this layer from all clients
            layer_weights = [client_weights[client_idx][layer_idx] for client_idx in range(self.num_clients)]

            # Compute the average (FedAvg algorithm)
            avg_layer_weight = np.mean(layer_weights, axis=0)
            averaged_weights.append(avg_layer_weight)

        # Update global model with FedAvg aggregated weights
        self.global_model.set_weights(averaged_weights)
        print("FedAvg aggregation completed")

    def evaluate_global_model(self, X_test, y_test):
        """Evaluate the global model on test data"""
        loss, accuracy = self.global_model.evaluate(X_test, y_test, verbose=0)
        return accuracy

    def train(self, X_train, y_train, X_test, y_test):
        """Main federated learning training loop with FedAvg"""
        print(f"\n=== Starting Federated Learning with {self.num_clients} Clients using FedAvg ===")

        # Train for multiple communication rounds
        for r in range(self.rounds):
            print(f"\n{'='*60}")
            print(f"FEDERATED ROUND {r+1}/{self.rounds} - FedAvg Aggregation")
            print(f"{'='*60}")

            # Phase 1: Distribute global model to all clients
            print(f"\nPhase 1: Distributing global model to {self.num_clients} clients...")
            for i in range(self.num_clients):
                self.client_models[i].set_weights(self.global_model.get_weights())

            # Phase 2: Local training on each client
            print(f"\nPhase 2: Local training on each client...")
            for i in range(self.num_clients):
                print(f"\n--- Training Client {i+1}/{self.num_clients} ---")
                self.client_models[i].fit(X_train, y_train, epochs=self.client_epochs, batch_size=32, verbose=0)

            # Phase 3: FedAvg Aggregation
            print(f"\nPhase 3: Applying FedAvg aggregation across {self.num_clients} clients...")
            self.fedavg_aggregate()

            # Phase 4: Evaluate global model
            print(f"\nPhase 4: Evaluating global model...")
            accuracy = self.evaluate_global_model(X_test, y_test)

            print(f"\nGlobal model accuracy after round {r+1}: {accuracy:.4f}")

        print(f"\n=== Federated Learning with {self.num_clients} Clients Completed ===")
        return self.global_model
