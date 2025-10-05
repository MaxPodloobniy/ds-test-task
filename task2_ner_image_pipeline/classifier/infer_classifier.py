import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses most warnings


def main():
    # Load the model
    MODEL_PATH = "models/classifier/best_classifier_model.keras"
    model = tf.keras.models.load_model(MODEL_PATH)

    # Classes used during training
    CLASS_NAMES = ['chimpanzee', 'coyote', 'deer', 'duck', 'eagle', 'elephant', 'hedgehog', 'kangaroo', 'rhinoceros',
                   'tiger']

    if len(sys.argv) < 2:
        print("Error: No image path provided!")
        sys.exit(1)

    image_path = sys.argv[1]

    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    print(f"Predicted class: {predicted_class}")


if __name__ == "__main__":
    main()
