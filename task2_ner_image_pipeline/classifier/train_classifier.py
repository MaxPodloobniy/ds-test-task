import os
import argparse
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split


def parse_args():
    """Function for parsing arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--test_split', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model_save_path', type=str, default='classifier_model.keras')

    return parser.parse_args()


def main():
    """Train image classification model using VGG16 transfer learning."""
    # Loading all arguments
    args = parse_args()

    print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

    # Get all file paths and class names
    all_files = []
    dataset_path = args.dataset_path
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                all_files.append((os.path.join(class_path, filename), class_name))

    # Create DataFrame
    df = pd.DataFrame(all_files, columns=["filename", "class"])

    # Split data into train and validation sets while preserving class distribution
    train_df, val_df = train_test_split(df, test_size=args.test_split, stratify=df["class"], random_state=42)

    # Define data generators
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filename",
        y_col="class",
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=True,
        seed=42,
        class_mode="categorical"
    )

    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filename",
        y_col="class",
        target_size=(args.img_size, args.img_size),
        batch_size=args.val_batch_size,
        shuffle=False,
        seed=42,
        class_mode="categorical"
    )

    print(f'Train generator classes: {train_generator.class_indices}')
    print(f'Validation generator classes: {valid_generator.class_indices}')

    # Save class indices for inference
    class_indices_path = args.model_save_path.replace('.keras', '_classes.json')
    with open(class_indices_path, 'w') as f:
        json.dump(train_generator.class_indices, f, indent=4)

    # Load VGG16 model without the fully connected layers
    base_model = tf.keras.applications.VGG16(
        input_shape=(args.img_size, args.img_size, 3),
        include_top=False,
        weights='imagenet'
    )

    # Fine-tune only the first 15 layers
    for layer in base_model.layers[:15]:
        layer.trainable = True
    for layer in base_model.layers[15:]:
        layer.trainable = False

    # Build the new model
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    num_classes = len(train_generator.class_indices)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    # Compile model
    model.compile(optimizer=Adam(args.learning_rate),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        args.model_save_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    early_stopping_callback = EarlyStopping(
        patience=4,
        restore_best_weights=True
    )

    learning_rate_reduce = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        mode='min'
    )

    # Train the model
    model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=valid_generator,
        callbacks=[checkpoint_callback, early_stopping_callback, learning_rate_reduce]
    )


if __name__ == "__main__":
    main()
