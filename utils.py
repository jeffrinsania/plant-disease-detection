from PIL import Image
import numpy as np

def load_and_preprocess_image(path, target_size=(224, 224)):
    img = Image.open(path).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img).astype('float32') / 255.0
    return arr

def preprocess_image_array(arr):
    # Model expects batch dimension
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
    return arr
