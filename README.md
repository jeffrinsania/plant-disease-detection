# AI-Powered Plant Disease Pre-Symptom Detection System

Quick setup and install instructions (Windows PowerShell)

1. Create a Python virtual environment and activate it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Upgrade pip and install requirements

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Prepare your dataset

- Organize images into a directory structure suitable for `image_dataset_from_directory`:

```
data/
  train/
    healthy/
    diseaseA/
    diseaseB/
  val/
    healthy/
    diseaseA/
    diseaseB/
```

4. Train a model (example)

```powershell
python train.py --data_dir data --epochs 10 --batch_size 32 --model_out saved_model
```

5. Run the Flask app for inference

```powershell
python app.py --model saved_model
```

Then open `http://127.0.0.1:5000/` and POST an image to `/predict` (multipart form `file`).

Notes:
- This repository contains simple starter scripts for training and inference with TensorFlow/Keras.
- For mobile deployment consider converting the saved model to TensorFlow Lite.