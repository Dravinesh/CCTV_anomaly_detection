import os
import cv2
import numpy as np
from tqdm import tqdm

dataset_path = "C:\\Users\\dravi\\Documents\\mini project\\datasets\\ucfdataset\\Train"
output_path = "C:\\Users\\dravi\\Documents\\mini project\\processed_data"
img_size = (128, 128)  # Reduce image size to save memory

os.makedirs(output_path, exist_ok=True)

CATEGORIES = os.listdir(dataset_path)  # Get category names

for category in CATEGORIES:
    category_path = os.path.join(dataset_path, category)
    if not os.path.exists(category_path):
        print(f"⚠️ Skipping {category} (not found)")
        continue
    
    images = []
    labels = []
    
    frames = sorted(os.listdir(category_path))
    frames = [f for f in frames if f.endswith(('.png', '.jpg', '.jpeg'))]  

    print(f"🚀 Processing {category} ({len(frames)} images)")
    
    for frame in tqdm(frames[:5000]):  # Limit to 5000 per category
        frame_path = os.path.join(category_path, frame)
        
        try:
            img = cv2.imread(frame_path)
            if img is None:
                print(f"⚠️ Skipping unreadable file: {frame_path}")
                continue  

            img = cv2.resize(img, img_size)
            img = img / 255.0  
            images.append(img)
            labels.append(CATEGORIES.index(category))

            # **Save every 1000 images to avoid high RAM usage**
            if len(images) % 1000 == 0:
                np.save(os.path.join(output_path, f"{category}_{len(images)}.npy"), np.array(images))
                np.save(os.path.join(output_path, f"{category}_labels_{len(labels)}.npy"), np.array(labels))
                images = []  # Clear memory
                labels = []

        except Exception as e:
            print(f"⚠️ Error reading {frame_path}: {e}")

    # Save remaining images
    if images:
        np.save(os.path.join(output_path, f"{category}_final.npy"), np.array(images))
        np.save(os.path.join(output_path, f"{category}_labels_final.npy"), np.array(labels))

print("✅ Data Processing Complete! Files saved in:", output_path)
