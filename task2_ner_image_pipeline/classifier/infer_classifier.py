import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


def parse_args():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description='Inference image classification model')

    parser.add_argument('--image_path', type=str, required=True, help='Path to image')
    parser.add_argument('--model_path', type=str, default='classifier_model.keras', help='Path to model')

    return parser.parse_args()


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses most warnings
    args = parse_args()

    # Load the model
    model = tf.keras.models.load_model(args.model_path)

    # Classes used during training
    CLASS_NAMES = ['chimpanzee', 'coyote', 'deer', 'duck', 'eagle', 'elephant', 'hedgehog', 'kangaroo', 'rhinoceros',
                   'tiger']

    img = image.load_img(args.image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    print(f"Predicted class: {predicted_class}")


if __name__ == "__main__":
    main()
