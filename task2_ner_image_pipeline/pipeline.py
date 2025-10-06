"""
Combined pipeline for text NER and image classification.

This script combines Named Entity Recognition (NER) for extracting animal names
from text with image classification to verify if the statement matches reality.
"""
import argparse
import sys
from classifier.infer_classifier import classify_image
from ner.infer_ner import extract_animals


def parse_args():
    """Parse command line arguments for pipeline"""
    parser = argparse.ArgumentParser(description='Pipeline with image classification and NER')

    parser.add_argument('--image_path', type=str, required=True, help='Path to image')
    parser.add_argument('--text', type=str, required=True, help='Text input')
    parser.add_argument('--ner_model_path', type=str, default='ner/model', help='Path to named entity recognition model')
    parser.add_argument('--class_model_path', type=str, default='classifier/model/classifier_model.keras', help='Path to classifier model')

    return parser.parse_args()


def main():
    args = parse_args()

    predicted_animal, confidence = classify_image(args.image_path, args.class_model_path)
    extracted_animals = extract_animals(args.text, args.ner_model_path)

    # Display extracted information
    print(f"Text: {args.text}")
    print(f"Extracted animals: {', '.join(extracted_animals) if extracted_animals else 'None'}")
    print(f"Image classification: {predicted_animal} (confidence: {confidence:.2%})")

    # Check if the classified animal matches any extracted from the text
    predicted_animal_lower = predicted_animal.lower()
    extracted_animals_lower = [a.lower() for a in extracted_animals]

    if extracted_animals_lower and predicted_animal_lower in extracted_animals_lower:
        print("✅ The statement is TRUE!")
        return 0
    else:
        print("❌ The statement is FALSE!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
