def resize_image(image, target_size):
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    image = load_img(image, target_size=target_size)
    return img_to_array(image)

def normalize_image(image):
    return image / 255.0

def augment_image(image):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen.flow(image)

def load_dataset(data_directory):
    import os
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(rescale=1.0/255.0)
    dataset = datagen.flow_from_directory(
        data_directory,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    return dataset

def save_model(model, model_path):
    model.save(model_path)

def load_model(model_path):
    from tensorflow.keras.models import load_model
    return load_model(model_path)