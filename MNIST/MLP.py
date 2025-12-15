import numpy as np

class NumpyHistory:
    def __init__(self, history_dict):
        self.history = history_dict

class NumpyMLP:
    def __init__(self, 
                 input_shape=(784,), 
                 hidden_units=[256,128], 
                 activation='relu',
                 loss='categorical_crossentropy',
                 metrics=['accuracy', 'precision', 'recall', 'TopKCategoricalAccuracy'],
                 dropout_rate=0.0, 
                 num_classes=10, 
                 learning_rate=0.01):
        
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.activation_name = activation
        self.loss_name = loss
        self.metrics = metrics
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weights, self.biases = self._initialize_weights()
        
        if activation == 'relu':
            self.activation = self._relu
            self.activation_derivative = self._relu_derivative
        elif activation == 'tanh':
            self.activation = np.tanh
            self.activation_derivative = lambda x: 1 - np.tanh(x)**2
        elif activation == 'sigmoid':
            self.activation = lambda x: 1/(1+np.exp(-x))
            self.activation_derivative = lambda x: self.activation(x)*(1-self.activation(x))
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        if loss == 'categorical_crossentropy':
            self.loss_fn = self._categorical_crossentropy
        elif loss == 'mse':
            self.loss_fn = self._mean_squared_error
        else:
            raise ValueError(f"Unknown loss function: {loss}")
    
    def _initialize_weights(self):
        layer_sizes = [self.input_shape[0]] + self.hidden_units + [self.num_classes]
        weights = []
        biases = []
        np.random.seed(42)
        for i in range(len(layer_sizes)-1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            weights.append(w)
            biases.append(b)
        return weights, biases
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def _softmax(self, x):
        exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    
    def _categorical_crossentropy(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-12, 1.0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def _mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    def _forward(self, X):
        activations = [X]
        zs = []
        for i in range(len(self.weights)-1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            a = self.activation(z)
            zs.append(z)
            activations.append(a)
        z_out = activations[-1] @ self.weights[-1] + self.biases[-1]
        a_out = self._softmax(z_out)
        zs.append(z_out)
        activations.append(a_out)
        return activations, zs
    
    def _backward(self, X, Y, activations, zs):
        m = X.shape[0]
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        delta = (activations[-1] - Y) / m
        grads_w[-1] = activations[-2].T @ delta
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True)

        for i in reversed(range(len(self.weights)-1)):
            delta = delta @ self.weights[i+1].T * self.activation_derivative(zs[i])
            grads_w[i] = activations[i].T @ delta
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]
    
    def _compute_metrics(self, Y_true, Y_pred, metrics=['accuracy'], top_k=5):
        results = {}
        y_true_labels = np.argmax(Y_true, axis=1)
        y_pred_labels = np.argmax(Y_pred, axis=1)
        
        if 'accuracy' in metrics:
            results['accuracy'] = np.mean(y_true_labels == y_pred_labels)
        
        if 'precision' in metrics:
            precisions = []
            for c in range(self.num_classes):
                tp = np.sum((y_pred_labels==c) & (y_true_labels==c))
                fp = np.sum((y_pred_labels==c) & (y_true_labels!=c))
                precisions.append(tp/(tp+fp+1e-12))
            results['precision'] = np.mean(precisions)
        
        if 'recall' in metrics:
            recalls = []
            for c in range(self.num_classes):
                tp = np.sum((y_pred_labels==c) & (y_true_labels==c))
                fn = np.sum((y_pred_labels!=c) & (y_true_labels==c))
                recalls.append(tp/(tp+fn+1e-12))
            results['recall'] = np.mean(recalls)
        
        if 'TopKCategoricalAccuracy' in metrics:
            top_k_preds = np.argsort(Y_pred, axis=1)[:, -top_k:]
            topk_hits = [y_true_labels[i] in top_k_preds[i] for i in range(len(y_true_labels))]
            results['TopKCategoricalAccuracy'] = np.mean(topk_hits)
        
        return results

    def fit(self, X_train, Y_train, validation_data=None, epochs=10, batch_size=32, metrics=['accuracy', 'precision', 'recall', 'TopKCategoricalAccuracy'], verbose=1):
        history = {metric: [] for metric in metrics}
        if validation_data:
            for metric in metrics:
                history['val_' + metric] = []
            history['val_loss'] = []
        history['loss'] = []

        for epoch in range(epochs):
            perm = np.random.permutation(X_train.shape[0])
            X_shuf, Y_shuf = X_train[perm], Y_train[perm]

            for start in range(0, X_train.shape[0], batch_size):
                end = start + batch_size
                X_batch, Y_batch = X_shuf[start:end], Y_shuf[start:end]
                activations, zs = self._forward(X_batch)
                self._backward(X_batch, Y_batch, activations, zs)

            # Compute train metrics & loss
            Y_pred_train = self.predict(X_train)
            train_metrics = self._compute_metrics(Y_train, Y_pred_train, metrics)
            for k, v in train_metrics.items():
                history[k].append(v)
            history['loss'].append(self.loss_fn(Y_train, Y_pred_train))

            # Validation
            if validation_data:
                X_val, Y_val = validation_data
                Y_pred_val = self.predict(X_val)
                val_metrics = self._compute_metrics(Y_val, Y_pred_val, metrics)
                for k, v in val_metrics.items():
                    history['val_' + k].append(v)
                history['val_loss'].append(self.loss_fn(Y_val, Y_pred_val))

            if verbose:
                msg = f"Epoch {epoch+1}/{epochs} | loss: {history['loss'][-1]:.4f}"
                msg += " | " + " | ".join([f"{k}: {history[k][-1]:.4f}" for k in self.metrics])
                if validation_data:
                    msg += f" | val_loss: {history['val_loss'][-1]:.4f}"
                    msg += " | " + " | ".join([f"val_{k}: {history['val_' + k][-1]:.4f}" for k in self.metrics])
                print(msg)

        return NumpyHistory(history)
    
    def evaluate(self, X_input, Y_true):
        predictions = self.predict(X_input)
        accuracy = np.mean(np.argmax(Y_true, axis=1) == np.argmax(predictions, axis=1))
        loss = -np.mean(np.sum(Y_true * np.log(np.clip(predictions, 1e-12, 1.0)), axis=1))
        return loss, accuracy

    def predict(self, X):
        activations, _ = self._forward(X)
        return activations[-1]
