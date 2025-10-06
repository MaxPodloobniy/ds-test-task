import argparse


def parse_args():
    """Parse command line arguments for pipeline"""
    parser = argparse.ArgumentParser(description='Pipeline with image classification and NER')

    parser.add_argument('--image_path', type=str, required=True, help='Path to image')
    parser.add_argument('--model_path', type=str, default='model/classifier_model.keras', help='Path to model')

    return parser.parse_args()


def main():
    # Ensure the correct number of command-line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python pipeline.py '<text>' <image_path>")
        sys.exit(1)

    input_text = sys.argv[1]
    image_path = sys.argv[2]

    # Extract animal names from text using NER model
    extracted_animals = run_ner(input_text)
    # Predict the class of the given image
    predicted_animal = run_classifier(image_path)

    # Display extracted information
    print(f"Text: {input_text}")
    print(f"Extracted animals: {', '.join(extracted_animals) if extracted_animals else 'None'}")
    print(f"Image classification: {predicted_animal}")

    # Check if the classified animal matches any extracted from the text
    if extracted_animals and predicted_animal in extracted_animals:
        print("✅ The statement is TRUE!")
    else:
        print("❌ The statement is FALSE!")


if __name__ == "__main__":
    main()
