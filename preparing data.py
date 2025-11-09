import numpy as np
import os

# Path to processed data
processed_data_path = "C:\\Users\\dravi\\Documents\\mini project\\processed_data"

# Define categories (same as dataset classes)
CATEGORIES = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 
              'Fighting', 'NormalVideos', 'RoadAccidents', 'Robbery', 
              'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']

def load_data():
    X_list = []
    y_list = []

    for category in CATEGORIES:
        feature_file = os.path.join(processed_data_path, f"{category}_1000.npy")
        label_file = os.path.join(processed_data_path, f"{category}_labels_1000.npy")

        if os.path.exists(feature_file) and os.path.exists(label_file):
            X_list.append(np.load(feature_file))
            y_list.append(np.load(label_file))
        else:
            print(f"⚠️ Missing files for category {category}, skipping...")

    # Concatenating all categories
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    return X, y

# Load processed dataset
X, y = load_data()

# Print dataset shape
print("✅ Data Loaded Successfully")
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
