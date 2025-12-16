# KerasMLP – szczegółowe omówienie kodu

---

## 1. Konstruktor `__init__`

```python
def __init__(self,
             input_shape=(784,),
             hidden_units=[256, 128],
             activation='relu',
             dropout_rate=0.3,
             num_classes=10,
             loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['loss', 'accuracy', 'precision', 'recall', 'TopKCategoricalAccuracy']):
    
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
```

* **Cel:** inicjalizacja modelu Keras MLP
* **Parametry:**

  * `input_shape` – liczba cech wejściowych
  * `hidden_units` – lista liczby neuronów w ukrytych warstwach
  * `activation` – funkcja aktywacji (`relu`, `tanh`, etc.)
  * `dropout_rate` – dropout po każdej warstwie ukrytej
  * `num_classes` – liczba klas wyjściowych
  * `loss` – funkcja straty (`categorical_crossentropy` lub `mse`)
  * `optimizer` – optymalizator Keras (`adam`, `sgd`, etc.)
  * `metrics` – lista metryk (`accuracy`, `precision`, `recall`, `TopKCategoricalAccuracy`)

---

## 2. Budowa modelu `_build_model`

```python
def _build_model(self):
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

    self._model.compile(optimizer='adam', loss=self.loss, metrics=keras_metrics)
```

* Sequential: warstwy gęste (`Dense`) + opcjonalny dropout
* Ostatnia warstwa: `softmax` dla klasyfikacji wieloklasowej
* Metryki Keras: `accuracy`, `Precision()`, `Recall()`, `TopKCategoricalAccuracy(k=10)`
* Kompilacja modelu (`optimizer`, `loss`, `metrics`)

---

## 3. Rozwiązywanie optymalizatora `_resolve_optimizer`

```python
def _resolve_optimizer(self):
    if isinstance(self.optimizer, str):
        opt_name = self.optimizer.lower()
        if opt_name == 'adam':
            return tf.keras.optimizers.Adam()
        elif opt_name == 'sgd':
            return tf.keras.optimizers.SGD()
        else:
            return tf.keras.optimizers.get(self.optimizer)
    return self.optimizer
```

* Zamienia string na obiekt Keras Optimizer
* Obsługuje `adam`, `sgd`, lub dowolny inny via `tf.keras.optimizers.get()`

---

## 4. Trening `fit`

```python
def fit(self, X_train, Y_train, validation_data=None, epochs=15, batch_size=128, callbacks=None, verbose=1):
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
```

* Wrapper dla `model.fit()` Keras
* Obsługa walidacji (`validation_data`)
* Obsługa callbacków Keras (np. EarlyStopping)
* Zwraca własny obiekt `History` kompatybilny z BaseModel

---

## 5. Predykcja `predict`

```python
def predict(self, X):
    return self._model.predict(X)
```

* Zwraca prawdopodobieństwa klas dla każdego przykładu
* Kształt wyniku: `(n_samples, num_classes)`

---

## 6. Ewaluacja `evaluate`

```python
def evaluate(self, X, Y):
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
```

* Oblicza stratę (loss) za pomocą `model.evaluate()`
* Dodatkowo liczy:

  * Accuracy
  * Precision (ważone)
  * Recall (ważone)
  * Top-K accuracy (`k=5`)

---

## 7. Kluczowe różnice względem NumpyMLP

* KerasMLP używa **backendu TensorFlow/Keras** – wszystkie operacje wektorowe są przyspieszone GPU/CPU
* Obsługuje **callbacki Keras** (np. EarlyStopping)
* Automatyczne liczenie gradientów i propagacja wstecz (`backprop`) w Keras
* Metryki dodatkowo obliczane ręcznie w `evaluate`, aby były zgodne z BaseModel i mogły uwzględniać np. ważone metryki tj. precyzja czy recall
