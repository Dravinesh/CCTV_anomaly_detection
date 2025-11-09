import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Load Model
model = tf.keras.models.load_model("best_lstm_model.keras")

# Load Test Data
X_test = np.load("C:/Users/dravi/Documents/mini project/processed_data/X_test.npy")
y_test = np.load("C:/Users/dravi/Documents/mini project/processed_data/y_test.npy")

# Print Shapes
print("✅ Model Expected Input Shape:", model.input_shape)
print("✅ X_test Shape:", X_test.shape)
print("✅ y_test Shape:", y_test.shape)
if len(X_test.shape) == 4 and model.input_shape[-1] != X_test.shape[-1]:
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
    print("✅ X_test Reshaped to:", X_test.shape)
print("Model Output Shape:", model.output_shape)
print("y_test Shape:", y_test.shape)
if len(y_test.shape) == 3:
    y_test = y_test[:, -1, :]  # Take only the last time step’s label
    print("✅ y_test Reshaped to:", y_test.shape)
y_pred = model.predict(X_test)
print("✅ Prediction Successful! y_pred Shape:", y_pred.shape)
from sklearn.metrics import accuracy_score

y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
y_test_classes = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"✅ Test Accuracy: {accuracy:.3f}")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification Report
print("🔍 Classification Report:\n", classification_report(y_true_classes, y_pred_classes))

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(y_test.shape[1]), yticklabels=range(y_test.shape[1]))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Plot Loss & Accuracy (Only if training history is saved)
history = np.load("C:/Users/dravi/Documents/mini project/processed_data/training_history.npy", allow_pickle=True).item()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['loss'], label="Train Loss")
plt.plot(history['val_loss'], label="Val Loss")
plt.legend()
plt.title("Loss Over Epochs")

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label="Train Accuracy")
plt.plot(history['val_accuracy'], label="Val Accuracy")
plt.legend()
plt.title("Accuracy Over Epochs")

plt.show()