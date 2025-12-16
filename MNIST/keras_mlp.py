import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from base_model import BaseModel, History

class KerasMLP(BaseModel):
    """
    Multi-Layer Perceptron using Keras backend.
    Fully compatible with BaseModel API for training, evaluation, and prediction.
    """
    def __init__(
        self,
        input_shape=(784,),
        hidden_units=[256, 128],
        activation='relu',
        dropout_rate=0.3,
        num_classes=10,
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['loss', 'accuracy', 'precision', 'recall', 'TopKCategoricalAccuracy']
    ):
        """
        Initialize Keras MLP model.

        Parameters
        ----------
        input_shape : tuple
            Shape of input data (e.g., (784,) for flattened MNIST).
        hidden_units : list[int]
            Number of neurons in each hidden layer.
        activation : str
            Activation function for hidden layers.
        dropout_rate : float
            Dropout rate after each hidden layer.
        num_classes : int
            Number of output classes.
        loss : str
            Loss function for training ('categorical_crossentropy' or 'mse').
        optimizer : str or tf.keras.optimizers.Optimizer
            Optimizer for training ('adam', 'sgd', etc.).
        metrics : list[str]
            List of metrics to track during training.
        """
        super().__init__()
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self._build_model()

    def _build_model(self):
        """
        Build and compile the internal Keras model.
        """
        self._model = Sequential()
        self._model.add(Dense(self.hidden_units[0], activation=self.activation,
                              input_shape=self.input_shape))
        if self.dropout_rate > 0:
            self._model.add(Dropout(self.dropout_rate))

        for units in self.hidden_units[1:]:
            self._model.add(Dense(units, activation=self.activation))
            if self.dropout_rate > 0:
                self._model.add(Dropout(self.dropout_rate))

        self._model.add(Dense(self.num_classes, activation='softmax'))

        keras_metrics = []
        for m in self.metrics:
            if m.lower() == 'accuracy':
                keras_metrics.append('accuracy')
            elif m.lower() == 'precision':
                keras_metrics.append(tf.keras.metrics.Precision())
            elif m.lower() == 'recall':
                keras_metrics.append(tf.keras.metrics.Recall())
            elif m.lower() == 'topkcategoricalaccuracy':
                keras_metrics.append(tf.keras.metrics.TopKCategoricalAccuracy(k=10))

        self._model.compile(optimizer='adam',
                            loss=self.loss,
                            metrics=keras_metrics)

    def _resolve_optimizer(self):
        """
        Resolve optimizer string to Keras optimizer object.
        """
        if isinstance(self.optimizer, str):
            opt_name = self.optimizer.lower()
            if opt_name == 'adam':
                return tf.keras.optimizers.Adam()
            elif opt_name == 'sgd':
                return tf.keras.optimizers.SGD()
            else:
                return tf.keras.optimizers.get(self.optimizer)
        return self.optimizer

    def fit(self, X_train, Y_train, validation_data=None, epochs=15, batch_size=128, callbacks=None, verbose=1):
        """
        Train the model on given data.

        Returns
        -------
        History object containing loss and metrics.
        """
        callbacks = callbacks or []
        history_obj = self._model.fit(
            X_train, Y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        return History(history_obj.history)

    def predict(self, X):
        """
        Predict class probabilities for input data.

        Returns
        -------
        ndarray
            Predicted probabilities of shape (n_samples, num_classes)
        """
        return self._model.predict(X)

    def evaluate(self, X, Y):
        """
        Evaluate the model on test data.

        Returns
        -------
        dict
            Dictionary with 'loss' and other metrics.
        """
        y_pred_probs = self.predict(X)
        y_pred_labels = np.argmax(y_pred_probs, axis=1)
        y_true_labels = np.argmax(Y, axis=1)

        results = {'loss': self._model.evaluate(X, Y, verbose=0)}

        if 'accuracy' in self.metrics:
            results['accuracy'] = np.mean(y_pred_labels == y_true_labels)

        if 'precision' in self.metrics:
            results['precision'] = precision_score(y_true_labels, y_pred_labels, average='weighted')

        if 'recall' in self.metrics:
            results['recall'] = recall_score(y_true_labels, y_pred_labels, average='weighted')

        if 'TopKCategoricalAccuracy' in self.metrics:
            k = 5
            topk = np.argsort(y_pred_probs, axis=1)[:, -k:]
            results['TopKCategoricalAccuracy'] = np.mean([y_true_labels[i] in topk[i] for i in range(len(y_true_labels))])

        return results
