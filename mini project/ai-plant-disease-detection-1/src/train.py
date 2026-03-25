import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import CNNModel

def load_data(data_dir, img_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_generator = datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    validation_generator = datagen.flow_from_directory(
        os.path.join(data_dir, 'validation'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    return train_generator, validation_generator

def train_model(train_generator, validation_generator, epochs=50):
    model = CNNModel()
    model.compile_model()
    model.fit(train_generator, validation_data=validation_generator, epochs=epochs)
    model.save('models/plant_disease_model.h5')

if __name__ == "__main__":
    data_directory = 'data/processed'
    train_gen, val_gen = load_data(data_directory)
    train_model(train_gen, val_gen)