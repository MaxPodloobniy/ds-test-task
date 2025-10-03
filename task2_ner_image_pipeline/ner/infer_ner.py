import spacy
import argparse

# Path to the custom model
CUSTOM_MODEL_PATH = "models/ner/custom_ner_model"

# List of animals we are looking for
animals = {"chimpanzee", "coyote", "deer", "duck", "eagle", "elephant", "hedgehog", "kangaroo", "rhinoceros", "tiger"}


def extract_animals(text):
    """Function to extract animals from text using ONLY the custom model."""
    nlp = spacy.load(CUSTOM_MODEL_PATH)

    # Process the text
    doc = nlp(text)

    # Select only entities labeled as "ANIMAL"
    found_animals = [ent.text.lower() for ent in doc.ents if ent.label_ == "ANIMAL"]

    return found_animals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER for animal extraction.")
    parser.add_argument("text", type=str, help="Input text for NER")
    args = parser.parse_args()

    animals_found = extract_animals(args.text)
    print("Found animals:", animals_found)
