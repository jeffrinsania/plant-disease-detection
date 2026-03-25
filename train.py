import argparse
import os
import json
import tensorflow as tf

def build_model(num_classes, input_shape=(224,224,3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def main(args):
    # Use image_dataset_from_directory for convenience
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')

    img_size = (224, 224)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_ds = None
    if os.path.isdir(val_dir):
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            val_dir,
            labels='inferred',
            label_mode='int',
            image_size=img_size,
            batch_size=args.batch_size,
            shuffle=False
        )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    model = build_model(num_classes, input_shape=(img_size[0], img_size[1], 3))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print('Training with classes:', class_names)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    # Save model
    os.makedirs(args.model_out, exist_ok=True)
    model_path = os.path.join(args.model_out, 'model')
    model.save(model_path)

    # Save class names
    with open(os.path.join(args.model_out, 'class_names.json'), 'w', encoding='utf-8') as f:
        json.dump(class_names, f)

    print('Saved model to', model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Directory with train/ and val/ subfolders')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_out', type=str, default='saved_model')
    args = parser.parse_args()
    main(args)
