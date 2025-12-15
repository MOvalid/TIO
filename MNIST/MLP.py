import numpy as np

class NumpyMLP:
    def __init__(
        self, 
        input_shape=(784,), 
        hidden_units=[256,128], 
        activation='relu', 
        dropout_rate=0.0, 
        num_classes=10, 
        learning_rate=0.01
    ):
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weights, self.biases = self._initialize_weights()
    
    # Initialize weights and biases
    def _initialize_weights(self):
        layer_sizes = [self.input_shape[0]] + self.hidden_units + [self.num_classes]
        weights_list = []
        biases_list = []
        np.random.seed(42)

        for i in range(len(layer_sizes)-1):
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            bias_vector = np.zeros((1, layer_sizes[i+1]))
            weights_list.append(weight_matrix)
            biases_list.append(bias_vector)

        return weights_list, biases_list
    
    # Activation functions
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def _softmax(self, x):
        exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    
    # Forward pass
    def _forward(self, input_data):
        activations_list = [input_data]
        linear_combinations = []
        
        for layer_idx in range(len(self.weights) - 1):
            z = activations_list[-1] @ self.weights[layer_idx] + self.biases[layer_idx]
            a = self._relu(z)
            linear_combinations.append(z)
            activations_list.append(a)
        
        # Output layer
        z_output = activations_list[-1] @ self.weights[-1] + self.biases[-1]
        a_output = self._softmax(z_output)
        linear_combinations.append(z_output)
        activations_list.append(a_output)
        
        return activations_list, linear_combinations
    
    # Backward pass
    def _backward(self, input_batch, y_true_batch, activations_list, linear_combinations):
        batch_size = input_batch.shape[0]
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]

        # Output layer
        delta_output = (activations_list[-1] - y_true_batch) / batch_size
        weight_gradients[-1] = activations_list[-2].T @ delta_output
        bias_gradients[-1] = np.sum(delta_output, axis=0, keepdims=True)

        # Hidden layers
        for layer_idx in reversed(range(len(self.weights) - 1)):
            delta_output = delta_output @ self.weights[layer_idx + 1].T * self._relu_derivative(linear_combinations[layer_idx])
            weight_gradients[layer_idx] = activations_list[layer_idx].T @ delta_output
            bias_gradients[layer_idx] = np.sum(delta_output, axis=0, keepdims=True)
        
        # Update weights and biases
        for layer_idx in range(len(self.weights)):
            self.weights[layer_idx] -= self.learning_rate * weight_gradients[layer_idx]
            self.biases[layer_idx] -= self.learning_rate * bias_gradients[layer_idx]
    
    # Train the model
    def fit(self, X_train, Y_train, validation_data=None, epochs=10, batch_size=32, verbose=1):
        history = {'train_accuracy': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            permutation = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[permutation]
            Y_shuffled = Y_train[permutation]

            for start_idx in range(0, X_train.shape[0], batch_size):
                end_idx = start_idx + batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                Y_batch = Y_shuffled[start_idx:end_idx]
                
                activations_list, linear_combinations = self._forward(X_batch)
                self._backward(X_batch, Y_batch, activations_list, linear_combinations)

            # Compute accuracy
            train_predictions = self.predict(X_train)
            train_acc = np.mean(np.argmax(Y_train, axis=1) == np.argmax(train_predictions, axis=1))
            history['train_accuracy'].append(train_acc)

            if validation_data:
                X_val, Y_val = validation_data
                val_predictions = self.predict(X_val)
                val_acc = np.mean(np.argmax(Y_val, axis=1) == np.argmax(val_predictions, axis=1))
                history['val_accuracy'].append(val_acc)

            if verbose:
                msg = f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f}"
                if validation_data:
                    msg += f" | Val Acc: {val_acc:.4f}"
                print(msg)

        return history
    
    # Predict class probabilities
    def predict(self, X_input):
        activations_list, _ = self._forward(X_input)
        return activations_list[-1]
    
    # Evaluate the model
    def evaluate(self, X_input, Y_true):
        predictions = self.predict(X_input)
        accuracy = np.mean(np.argmax(Y_true, axis=1) == np.argmax(predictions, axis=1))
        loss = -np.mean(np.sum(Y_true * np.log(np.clip(predictions, 1e-12, 1.0)), axis=1))
        return loss, accuracy
