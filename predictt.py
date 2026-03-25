import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Load class names
class_names = np.load("class_names.npy")

# Image path (CHANGE if needed)
image_path = r"C:\Users\dell\Documents\Campus_Crowd_Analysis\test_leaf.jpg"

# Read image
img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found. Check file path.")
    exit()

# Resize
img = cv2.resize(img, (128,128))

# Normalize
img = img / 255.0

# Reshape
img = np.reshape(img, (1,128,128,3))

# Predict
prediction = model.predict(img)

predicted_class = np.argmax(prediction)

# Show disease name
print("Predicted Disease:", class_names[predicted_class])