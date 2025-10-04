import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def main():
    print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

    # Get all file paths and class names
    all_files = []
    dataset_path = '/Users/maxim/PycharmProjects/ImageClassification+NER/dataset'
    for class_name in os.listdir(dataset_path):
        if class_name.lower() != "hippopotamus":  # Exclude the "hippopotamus" class
            class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    all_files.append((os.path.join(class_path, filename), class_name))

    # Create DataFrame
    df = pd.DataFrame(all_files, columns=["filename", "class"])

    # Split data into train and validation sets while preserving class distribution
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["class"], random_state=42)

    # Define data generators
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filename",
        y_col="class",
        target_size=(224, 224),
        batch_size=24,
        shuffle=True,
        seed=42,
        class_mode="categorical"
    )

    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filename",
        y_col="class",
        target_size=(224, 224),
        batch_size=2,
        shuffle=True,
        seed=42,
        class_mode="categorical"
    )

    print(f'Train generator classes: {train_generator.class_indices}')
    print(f'Validation generator classes: {valid_generator.class_indices}')

    # Check min and max pixel values
    print(f'Min pixel value: {tf.reduce_min(train_generator[0][0])}')
    print(f'Max pixel value: {tf.reduce_max(train_generator[0][0])}')

    # Display sample images
    x_batch, y_batch = next(train_generator)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    for img, lbl, ax in zip(x_batch[:9], y_batch[:9], axes):
        ax.imshow(img)
        ax.set_title(f"Class: {np.argmax(lbl)}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # Load VGG16 model without the fully connected layers
    base_model = tf.keras.applications.VGG16(
        input_shape=(224, 224, 3),
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
    predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    # Compile model
    model.compile(optimizer=Adam(1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        "best_classifier_model.keras",
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
        epochs=100,
        validation_data=valid_generator,
        callbacks=[checkpoint_callback, early_stopping_callback, learning_rate_reduce]
    )


if __name__ == "__main__":
    main()
