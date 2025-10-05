import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array


def parse_args():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description='Inference image classification model')

    parser.add_argument('--image_path', type=str, required=True, help='Path to image')
    parser.add_argument('--model_path', type=str, default='classifier_model.keras', help='Path to model')

    return parser.parse_args()


def preprocess_image(image_path, target_size):
    """Load and preprocess image for model inference."""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def main():
    args = parse_args()

    # Check if image exists
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    # Check if model exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    # Check if classes exists
    if not os.path.exists(args.model_path.replace('.keras', '_classes.json')):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    # Load class names
    classes_path = args.model_path.replace('.keras', '_classes.json')
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Class names file not found: {classes_path}")

    with open(classes_path, 'r') as f:
        class_indices = json.load(f)

    # Invert dict: {class_name: idx} -> {idx: class_name}
    CLASS_NAMES = {v: k for k, v in class_indices.items()}

    # Load the model
    model = tf.keras.models.load_model(args.model_path)

    input_shape = model.input_shape[1:3]  # (height, width)
    img_array = preprocess_image(args.image_path, input_shape)

    predictions = model.predict(img_array, verbose=0)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    print(f"Predicted class: {predicted_class} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()
