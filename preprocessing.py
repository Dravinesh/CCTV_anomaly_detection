import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 🔹 Load the processed .npy files
categories = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 
              'Explosion', 'Fighting', 'NormalVideos', 'RoadAccidents', 
              'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']

X = []
y = []

data_path = "C:\\Users\\dravi\\Documents\\mini project\\processed_data\\"  # Change this path if needed

# 🔹 Load each category's data
for category in categories:
    try:
        x_data = np.load(data_path + f"{category}_1000.npy")
        y_data = np.load(data_path + f"{category}_labels_1000.npy")

        X.append(x_data)
        y.append(y_data)

        print(f"✅ Loaded {category} - {x_data.shape}")
    except FileNotFoundError:
        print(f"⚠️ Warning: {category} data not found, skipping...")

# 🔹 Convert lists to numpy arrays
X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)

print(f"\n✅ Data Loaded Successfully - Shape of X: {X.shape}, Shape of y: {y.shape}")

# 🔹 Normalize the images (convert pixel values from 0-255 to 0-1)
X = X.astype("float32") / 255.0
print("✅ Data Normalized")

# 🔹 Encode labels using one-hot encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Convert category names to numbers
y = to_categorical(y)  # Convert numbers to one-hot encoding
print("✅ Labels Encoded - Shape of y:", y.shape)

# 🔹 Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n✅ Data Split: Train - {X_train.shape}, Test - {X_test.shape}")

# 🔹 Save preprocessed data for future use
np.save(data_path + "X_train.npy", X_train)
np.save(data_path + "X_test.npy", X_test)
np.save(data_path + "y_train.npy", y_train)
np.save(data_path + "y_test.npy", y_test)

print("\n✅ Preprocessed Data Saved Successfully!")
