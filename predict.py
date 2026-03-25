import argparse
import json
import os
import numpy as np
import tensorflow as tf
from utils import load_and_preprocess_image, preprocess_image_array

def load_model_and_classes(model_dir):
    model_path = os.path.join(model_dir, 'model')
    model = tf.keras.models.load_model(model_path)
    with open(os.path.join(model_dir, 'class_names.json'), 'r', encoding='utf-8') as f:
        class_names = json.load(f)
    return model, class_names

def predict_image(model, class_names, image_path):
    arr = load_and_preprocess_image(image_path, target_size=(224,224))
    arr = preprocess_image_array(arr)
    preds = model.predict(arr)
    idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return {'label': class_names[idx], 'confidence': confidence}

def main(args):
    model, class_names = load_model_and_classes(args.model)
    res = predict_image(model, class_names, args.image)
    print(res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='saved_model', help='Path to model output folder')
    parser.add_argument('--image', type=str, required=True, help='Path to image to predict')
    args = parser.parse_args()
    main(args)
