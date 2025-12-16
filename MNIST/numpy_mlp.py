import numpy as np

from base_model import BaseModel, History
from metrics.classification import (
    accuracy,
    precision_weighted,
    recall_weighted,
    top_k_categorical_accuracy
)

class NumpyMLP(BaseModel):
    """
    Multi-Layer Perceptron implemented in pure NumPy.

    This class provides a fully functional neural network model with an API
    compatible with BaseModel, inspired by Keras-style workflows.
    It supports configurable activation functions, loss functions,
    metrics, callbacks, and training history tracking.
    """

    def __init__(
        self,
        input_shape=(784,),
        hidden_units=[256, 128],
        activation='relu',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        num_classes=10,
        learning_rate=0.01,
        dropout_rate=0.0 
    ):
        """
        Initialize the NumPy-based MLP model.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input feature vector.
        hidden_units : list
            Number of neurons in each hidden layer.
        activation : str
            Activation function name ('relu', 'tanh', 'sigmoid').
        loss : str
            Loss function name ('categorical_crossentropy', 'mse').
        metrics : list
            List of metric names to compute during training.
        num_classes : int
            Number of output classes.
        learning_rate : float
            Learning rate for gradient descent.
        """
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.activation_name = activation
        self.loss_name = loss
        self.metrics = metrics
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        self.weights, self.biases = self._init_weights()
        self.activation, self.activation_derivative = self._resolve_activation()
        self.loss_fn = self._resolve_loss()

    def _init_weights(self):
        """
        Initialize network weights and biases using He initialization.

        Returns
        -------
        tuple
            Lists of weight matrices and bias vectors.
        """
        layer_sizes = [self.input_shape[0]] + self.hidden_units + [self.num_classes]
        weights, biases = [], []
        np.random.seed(42)

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            weights.append(w)
            biases.append(b)

        return weights, biases

    def _resolve_activation(self):
        """
        Resolve the activation function and its derivative based on configuration.

        Returns
        -------
        tuple
            Activation function and its derivative.
        """
        if self.activation_name == 'relu':
            return self._relu, self._relu_derivative
        if self.activation_name == 'tanh':
            return np.tanh, lambda x: 1 - np.tanh(x) ** 2
        if self.activation_name == 'sigmoid':
            f = lambda x: 1 / (1 + np.exp(-x))
            return f, lambda x: f(x) * (1 - f(x))
        raise ValueError(f"Unknown activation: {self.activation_name}")

    def _resolve_loss(self):
        """
        Resolve the loss function based on configuration.

        Returns
        -------
        callable
            Loss function.
        """
        if self.loss_name == 'categorical_crossentropy':
            return self._categorical_crossentropy
        if self.loss_name == 'mse':
            return self._mean_squared_error
        raise ValueError(f"Unknown loss: {self.loss_name}")

    def _relu(self, x):
        """
        ReLU activation function.
        """
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        """
        Derivative of ReLU activation.
        """
        return (x > 0).astype(float)

    def _softmax(self, x):
        """
        Softmax activation function for output layer.
        """
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def _categorical_crossentropy(self, y_true, y_pred):
        """
        Compute categorical cross-entropy loss.
        """
        y_pred = np.clip(y_pred, 1e-12, 1.0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def _mean_squared_error(self, y_true, y_pred):
        """
        Compute mean squared error loss.
        """
        return np.mean((y_true - y_pred) ** 2)

    def _forward(self, X, training=True):
        """
        Perform forward propagation through the network.

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        tuple
            Activations and linear combinations (z-values).
        """
        activations, zs = [X], []

        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            a = self.activation(z)

            if training and self.dropout_rate > 0:
                mask = (np.random.rand(*a.shape) >= self.dropout_rate).astype(float)
                a *= mask
                a /= (1.0 - self.dropout_rate)

            zs.append(z)
            activations.append(a)

        z_out = activations[-1] @ self.weights[-1] + self.biases[-1]
        a_out = self._softmax(z_out)
        zs.append(z_out)
        activations.append(a_out)

        return activations, zs

    def _backward(self, X, Y, activations, zs):
        """
        Perform backward propagation and update weights.

        Parameters
        ----------
        X : ndarray
            Input batch.
        Y : ndarray
            True labels.
        activations : list
            Activations from forward pass.
        zs : list
            Linear combinations from forward pass.
        """
        m = X.shape[0]
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        delta = (activations[-1] - Y) / m
        grads_w[-1] = activations[-2].T @ delta
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True)

        for i in reversed(range(len(self.weights) - 1)):
            delta = delta @ self.weights[i + 1].T * self.activation_derivative(zs[i])
            grads_w[i] = activations[i].T @ delta
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def _compute_metrics(self, Y_true, Y_pred):
        """
        Compute all configured metrics.

        Returns
        -------
        dict
            Dictionary of metric values.
        """
        results = {}

        if 'accuracy' in self.metrics:
            results['accuracy'] = accuracy(Y_true, Y_pred)

        if 'precision' in self.metrics:
            results['precision'] = precision_weighted(Y_true, Y_pred, self.num_classes)

        if 'recall' in self.metrics:
            results['recall'] = recall_weighted(Y_true, Y_pred, self.num_classes)

        if 'TopKCategoricalAccuracy' in self.metrics:
            results['TopKCategoricalAccuracy'] = top_k_categorical_accuracy(Y_true, Y_pred)

        return results

    def _initialize_history(self, validation_data):
        """
        Initialize history dictionary for training.

        Returns
        -------
        dict
            Empty history structure.
        """
        history = {m: [] for m in self.metrics}
        history['loss'] = []

        if validation_data:
            for m in self.metrics:
                history[f'val_{m}'] = []
            history['val_loss'] = []

        return history

    def _train_single_epoch(self, X, Y, batch_size):
        """
        Train the model for a single epoch using mini-batches.
        """
        perm = np.random.permutation(X.shape[0])
        X_shuf, Y_shuf = X[perm], Y[perm]

        for i in range(0, X_shuf.shape[0], batch_size):
            X_batch = X_shuf[i:i + batch_size]
            Y_batch = Y_shuf[i:i + batch_size]

            activations, zs = self._forward(X_batch)
            self._backward(X_batch, Y_batch, activations, zs)

    def _evaluate_dataset(self, X, Y):
        """
        Evaluate loss and metrics on a dataset.
        """
        Y_pred = self.predict(X)
        return self.loss_fn(Y, Y_pred), self._compute_metrics(Y, Y_pred)

    def _log_results(self, history, loss, metrics, prefix=""):
        """
        Append evaluation results to history.
        """
        history[f'{prefix}loss'].append(loss)
        for k, v in metrics.items():
            history[f'{prefix}{k}'].append(v)

    def _run_callbacks(self, callbacks, epoch, logs):
        """
        Execute callbacks at the end of an epoch.
        """
        for cb in callbacks:
            cb.on_epoch_end(epoch, logs)
        return any(getattr(cb, 'stop_training', False) for cb in callbacks)

    def _print_epoch_summary(self, epoch, epochs, logs):
        """
        Print formatted epoch summary.
        """
        msg = f"Epoch {epoch + 1}/{epochs} | "
        msg += " | ".join(f"{k}: {v:.4f}" for k, v in logs.items())
        print(msg)

    def fit(
        self,
        X_train,
        Y_train,
        validation_data=None,
        epochs=10,
        batch_size=32,
        callbacks=None,
        verbose=1
    ):
        """
        Train the model.

        Returns
        -------
        History
            Training history object.
        """
        callbacks = callbacks or []
        history = self._initialize_history(validation_data)

        for cb in callbacks:
            cb.on_train_begin()

        for epoch in range(epochs):
            self._train_single_epoch(X_train, Y_train, batch_size)

            train_loss, train_metrics = self._evaluate_dataset(X_train, Y_train)
            self._log_results(history, train_loss, train_metrics)

            logs = {'loss': train_loss, **train_metrics}

            if validation_data:
                X_val, Y_val = validation_data
                val_loss, val_metrics = self._evaluate_dataset(X_val, Y_val)
                self._log_results(history, val_loss, val_metrics, prefix='val_')
                logs.update({f'val_{k}': v for k, v in val_metrics.items()})
                logs['val_loss'] = val_loss

            if self._run_callbacks(callbacks, epoch, logs):
                break

            if verbose:
                self._print_epoch_summary(epoch, epochs, logs)

        for cb in callbacks:
            cb.on_train_end()

        return History(history)

    def predict(self, X):
        """
        Predict class probabilities for input samples.
        """
        return self._forward(X)[0][-1]

    def evaluate(self, X, Y):
        """
        Evaluate the model on a dataset.

        Returns
        -------
        dict
            Dictionary containing loss and metrics.
        """
        Y_pred = self.predict(X)
        return {'loss': self.loss_fn(Y, Y_pred), **self._compute_metrics(Y, Y_pred)}
