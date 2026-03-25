import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf

# Load model and class names
model = tf.keras.models.load_model("plant_disease_model.h5")
class_names = np.load("class_names.npy")

# Window
root = tk.Tk()
root.title("AI Plant Disease Detector")
root.geometry("700x700")
root.configure(bg="#f1f8f4")

# ===== Header =====
header = tk.Frame(root, bg="#1b5e20", height=60)
header.pack(fill="x")

title = tk.Label(
    header,
    text="🌿 AI Plant Disease Detection",
    font=("Helvetica", 18, "bold"),
    bg="#1b5e20",
    fg="white"
)
title.pack(pady=10)

# ===== Card Frame =====
card = tk.Frame(root, bg="white", bd=2, relief="ridge")
card.pack(pady=30, padx=30, fill="both", expand=True)

# Subtitle
subtitle = tk.Label(
    card,
    text="Upload a leaf image to detect disease",
    font=("Arial", 12),
    bg="white",
    fg="#555"
)
subtitle.pack(pady=10)

# Image display box
img_frame = tk.Frame(card, bg="#eeeeee", width=300, height=300)
img_frame.pack(pady=15)

img_label = tk.Label(img_frame, bg="#eeeeee")
img_label.place(relx=0.5, rely=0.5, anchor="center")

# Result label
result_label = tk.Label(
    card,
    text="No image selected",
    font=("Arial", 14, "bold"),
    bg="white",
    fg="#2e7d32"
)
result_label.pack(pady=10)

# Confidence label
confidence_label = tk.Label(
    card,
    text="",
    font=("Arial", 11),
    bg="white",
    fg="#666"
)
confidence_label.pack()

# ===== Upload Function =====
def upload_image():
    file_path = filedialog.askopenfilename()

    if not file_path:
        return

    # Show image
    img = Image.open(file_path)
    img = img.resize((250, 250))
    img_display = ImageTk.PhotoImage(img)

    img_label.config(image=img_display)
    img_label.image = img_display

    # Process image
    img_cv = cv2.imread(file_path)
    img_cv = cv2.resize(img_cv, (128,128))
    img_cv = img_cv / 255.0
    img_cv = np.reshape(img_cv, (1,128,128,3))

    # Predict
    prediction = model.predict(img_cv)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    disease = class_names[predicted_class]

    # Update result
    result_label.config(
        text=f"🌱 {disease}",
        fg="#d32f2f" if "blight" in disease.lower() else "#2e7d32"
    )

    confidence_label.config(
        text=f"Confidence: {confidence:.2f}%"
    )

# ===== Button Hover Effects =====
def on_enter(e):
    upload_btn.config(bg="#2e7d32")

def on_leave(e):
    upload_btn.config(bg="#43a047")

# Upload Button
upload_btn = tk.Button(
    card,
    text="📂 Upload Image",
    command=upload_image,
    font=("Arial", 13, "bold"),
    bg="#43a047",
    fg="white",
    padx=15,
    pady=8,
    relief="flat",
    cursor="hand2"
)
upload_btn.pack(pady=20)

upload_btn.bind("<Enter>", on_enter)
upload_btn.bind("<Leave>", on_leave)

# Footer
footer = tk.Label(
    root,
    text="AI Mini Project • Plant Disease Detection System",
    font=("Arial", 10),
    bg="#f1f8f4",
    fg="gray"
)
footer.pack(pady=10)

# Run
root.mainloop()