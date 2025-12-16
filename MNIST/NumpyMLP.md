# NumpyMLP – szczegółowe omówienie kodu

---

## 1. Konstruktor `__init__`

```python
def __init__(self,
             input_shape=(784,),
             hidden_units=[256,128],
             activation='relu',
             loss='categorical_crossentropy',
             metrics=['accuracy'],
             num_classes=10,
             learning_rate=0.01,
             dropout_rate=0.0):
    
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
```

* **Cel:** inicjalizacja sieci, konfiguracja metryk, funkcji aktywacji i straty
* **Parametry:**

  * `input_shape` – liczba cech wejściowych (np. 784 dla MNIST)
  * `hidden_units` – lista liczby neuronów w warstwach ukrytych
  * `activation` – funkcja aktywacji w warstwach ukrytych (ReLU/Tanh/Sigmoid)
  * `dropout_rate` – dropout dla regularizacji
  * `num_classes` – liczba klas wyjściowych
  * `learning_rate` – krok uczenia
  * `loss` – funkcja straty (`categorical_crossentropy` lub `mse`)
  * `metrics` – lista metryk do monitorowania (`accuracy`, `precision`, `recall`, `TopKCategoricalAccuracy`)

---

## 2. Inicjalizacja wag `_init_weights`

```python
def _init_weights(self):
    layer_sizes = [self.input_shape[0]] + self.hidden_units + [self.num_classes]
    weights, biases = [], []
    np.random.seed(42)

    for i in range(len(layer_sizes) - 1):
        w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i])
        b = np.zeros((1, layer_sizes[i + 1]))
        weights.append(w)
        biases.append(b)

    return weights, biases
```

* **He initialization** dla ReLU (`np.sqrt(2 / n_in)`)
* Biasy inicjalizowane zerami
* Zwraca listy wag i biasów dla wszystkich warstw

---

## 3. Funkcje aktywacji

```python
def _relu(self, x):
    return np.maximum(0, x)

def _relu_derivative(self, x):
    return (x > 0).astype(float)

def _softmax(self, x):
    exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
```

* `_relu(x)` – aktywacja ReLU
* `_relu_derivative(x)` – pochodna ReLU (do backprop)
* `_softmax(x)` – konwersja wyjść na prawdopodobieństwa klas

---

## 4. Propagacja w przód `_forward`

* Oblicza **kombinacje liniowe `z`** i **aktywacje `a`** każdej warstwy
* `zs` – lista wszystkich wartości `z` (potrzebna w backprop)
* `activations` – lista aktywacji od wejścia do wyjścia
* Dropout opcjonalnie podczas treningu
* Ostatnia warstwa: `a_out = softmax(z_out)`

---

## 5. Backward pass `_backward`

```python
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
```

* Delta w warstwie wyjściowej: `(y_pred - y_true)/m`
* Gradienty wag: `a_prev.T @ delta`
* Gradienty biasów: `sum(delta)`
* Aktualizacja wag i biasów: `W -= lr * grad_W`, `b -= lr * grad_b`

---

## 6. Obliczanie metryk `_compute_metrics`

```python
def _compute_metrics(self, Y_true, Y_pred):
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
```

* Obsługuje wszystkie metryki podane w `self.metrics`
* Zwraca słownik `{ 'accuracy': ..., 'precision': ..., ... }`

---

## 7. Trening `fit`

* Mini-batch SGD z permutacją danych
* Forward + backward dla każdej partii
* Obliczenie metryk dla treningu i walidacji
* Obsługa callbacków (`on_epoch_end`, `stop_training`)
* Zwraca `History`:

```python
history = History(history_dict)
```

---

## 8. Predykcja `predict`

* Forward pass
* Zwraca prawdopodobieństwa dla każdej klasy:

```python
def predict(self, X):
    return self._forward(X)[0][-1]
```

---

## 9. Ewaluacja `evaluate`

* Oblicza stratę i wszystkie metryki skonfigurowane w `self.metrics`:

```python
def evaluate(self, X, Y):
    Y_pred = self.predict(X)
    return {'loss': self.loss_fn(Y, Y_pred), **self._compute_metrics(Y, Y_pred)}
```
