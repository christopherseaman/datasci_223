# Hints

Use these hints if you get stuck. Try to work through the problem yourself first!

---

## Part 1 Hints

### Building a Sequential model

```python
model_dense = Sequential([
    Input(shape=(32, 32, 3)),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(10, activation="softmax"),
])
```

### Compiling

```python
model_dense.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
```

### Training with EarlyStopping

```python
early_stop = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

history_dense = model_dense.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop],
)
```

### Evaluating and getting predictions

```python
test_loss, test_acc = model_dense.evaluate(X_test, y_test, verbose=0)

y_pred = np.argmax(model_dense.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
```

---

## Part 2 Hints

### Building a CNN

```python
model_cnn = Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax"),
])
```

### Training with multiple callbacks

```python
callbacks = [
    EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    ),
    ModelCheckpoint(
        "output/best_cnn.keras",
        save_best_only=True,
        monitor="val_accuracy",
    ),
]

history_cnn = model_cnn.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks,
)
```

### Plotting training history

```python
plot_training_history(
    history_cnn, os.path.join(OUTPUT_DIR, "part2_training_history.png")
)
```

---

## Part 3 Hints

### Building an LSTM

```python
model_lstm = Sequential([
    Input(shape=(140, 1)),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(5, activation="softmax"),
])
```

### Training the LSTM

```python
early_stop = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

history_lstm = model_lstm.fit(
    X_train_ecg, y_train_ecg,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
)
```

### ECG evaluation

```python
lstm_loss, lstm_acc = model_lstm.evaluate(X_test_ecg, y_test_ecg, verbose=0)

y_pred_ecg = np.argmax(model_lstm.predict(X_test_ecg, verbose=0), axis=1)
y_true_ecg = np.argmax(y_test_ecg, axis=1)
cm_ecg = confusion_matrix(y_true_ecg, y_pred_ecg)
```

---

## General Tips

### If training is slow
- Reduce epochs or increase batch_size
- Dense models train fastest; LSTMs are slowest

### If accuracy is low
- Make sure data is normalized (helpers do this for you)
- Check that you're using the right activation functions (relu for hidden, softmax for output)
- Try adding more units or layers
