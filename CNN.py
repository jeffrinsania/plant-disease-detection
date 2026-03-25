import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras import layers, models

# Dataset path
dataset_path = r"C:\Users\dell\Downloads\archive (1)\PlantVillage\PlantVillage"

images = []
labels = []

MAX_IMAGES = 5000   # limit to avoid memory crash

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

        img = cv2.resize(img, (128,128))

        images.append(img)
        labels.append(folder)

        if len(images) >= MAX_IMAGES:
            break

    if len(images) >= MAX_IMAGES:
        break

print("Total Images Loaded:", len(images))

# Convert to arrays
X = np.array(images, dtype=np.float32)
y = np.array(labels)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# 🔥 IMPORTANT LINE (fix your error)
np.save("class_names.npy", encoder.classes_)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training images:", len(X_train))
print("Testing images:", len(X_test))

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dense(len(set(y)), activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
print("Starting CNN training...")

model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

# Save model
model.save("plant_disease_model.h5")

print("Model saved successfully!")