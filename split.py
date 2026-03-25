import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

dataset_path = r"C:\Users\dell\Downloads\archive (1)\PlantVillage\PlantVillage"

images = []
labels = []

# Load images
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (224,224))

        images.append(img)
        labels.append(folder)

print("Total Images Loaded:", len(images))

# Convert to arrays
X = np.array(images)
y = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training images:", len(X_train))
print("Testing images:", len(X_test))