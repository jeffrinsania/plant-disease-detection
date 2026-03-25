import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from utils import preprocess_image

def load_trained_model(model_path):
    if os.path.exists(model_path):
        model = load_model(model_path)
        return model
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

def predict_disease(model, img_path):
    img = preprocess_image(img_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    predictions = model.predict(img)
    return predictions

def main(img_path, model_path):
    model = load_trained_model(model_path)
    predictions = predict_disease(model, img_path)
    print("Predictions:", predictions)

if __name__ == "__main__":
    # Example usage
    model_path = 'models/trained_model.h5'  # Update with the actual model path
    img_path = 'data/raw/sample_leaf.jpg'    # Update with the actual image path
    main(img_path, model_path)