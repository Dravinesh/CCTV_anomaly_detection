import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Load preprocessed data
X = np.load("C:/Users/dravi/Documents/mini project/processed_data/X_train.npy")  # Shape: (samples, 128, 128, 3)
y = np.load("C:/Users/dravi/Documents/mini project/processed_data/y_train.npy")  # Shape: (samples, categories)

# Flatten each 128x128 RGB image into a sequence of 128 timesteps with 128x3 features per step
X = X.reshape(X.shape[0], 128, 128 * 3)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check new shapes
print("✅ X_train shape:", X_train.shape)  # Expected: (samples, 128, 384)
print("✅ y_train shape:", y_train.shape)  # Expected: (samples, categories)

# Build LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(128, 128 * 3)),  # 128 timesteps, 384 features
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(y_train.shape[1], activation='softmax')  # Output layer for classification
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define checkpoint to save the best model
checkpoint = ModelCheckpoint("best_lstm_model.keras", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Save final model
model.save("final_lstm_model.keras")

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc:.4f}")
