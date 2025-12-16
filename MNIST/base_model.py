from abc import ABC, abstractmethod
import tensorflow as tf


class History:
    """
    Keras-compatible History object.
    Stores training and validation metrics per epoch.
    """
    def __init__(self, history_dict):
        self.history = history_dict


class Callback:
    """
    Base callback class.
    All callbacks should inherit from this class.
    """
    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.
    """
    def __init__(self, monitor='val_loss', patience=5, mode='min'):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.wait = 0
        self.stop_training = False

    def on_epoch_end(self, epoch, logs=None):
        value = logs.get(self.monitor)
        if value is None:
            return

        improved = (
            value < self.best if self.mode == 'min'
            else value > self.best
        )

        if improved:
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True


class KerasEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', patience=5, mode='min'):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is None:
            return
        improved = (value < self.best) if self.mode == 'min' else (value > self.best)
        if improved:
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

class BaseModel(ABC):
    """
    Abstract base class for all models (NumPy, Keras, etc.).
    Provides unified API similar to Keras.
    """

    @abstractmethod
    def fit(self, X_train, Y_train, validation_data=None,
            epochs=1, batch_size=32, callbacks=None, verbose=1):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def evaluate(self, X, Y):
        pass
