import os
import cv2

dataset_path = r"C:\Users\dell\Downloads\archive (1)\PlantVillage\PlantVillage"

images = []
labels = []

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