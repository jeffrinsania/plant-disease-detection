import argparse
import io
import json
import os
from flask import Flask, request, jsonify
import tensorflow as tf
from utils import load_and_preprocess_image, preprocess_image_array

app = Flask(__name__)
MODEL = None
CLASS_NAMES = None

def load_model(model_dir):
    model_path = os.path.join(model_dir, 'model')
    model = tf.keras.models.load_model(model_path)
    with open(os.path.join(model_dir, 'class_names.json'), 'r', encoding='utf-8') as f:
        class_names = json.load(f)
    return model, class_names

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400
    file = request.files['file']
    img_bytes = io.BytesIO(file.read())
    # Temporarily save to disk to reuse utils loader
    temp_path = 'temp_upload.jpg'
    with open(temp_path, 'wb') as f:
        f.write(img_bytes.getvalue())

    arr = load_and_preprocess_image(temp_path, target_size=(224,224))
    arr = preprocess_image_array(arr)
    preds = MODEL.predict(arr)
    import numpy as np
    idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    label = CLASS_NAMES[idx]

    # Basic preventive recommendations (placeholder)
    recommendations = {
        'healthy': 'No action required. Monitor regularly.',
        'default': 'Isolate affected plants and consider fungicide/pesticide based on expert advice.'
    }
    rec = recommendations.get(label.lower(), recommendations['default'])

    return jsonify({'label': label, 'confidence': confidence, 'recommendation': rec})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='saved_model', help='Path to model output folder')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    MODEL, CLASS_NAMES = load_model(args.model)
    app.run(host=args.host, port=args.port)
