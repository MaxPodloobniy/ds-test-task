"""
Named Entity Recognition (NER) inference script for extracting animal names.

This script loads a trained spaCy model and identifies entities labeled as "ANIMAL"
within a given text input.
"""
import argparse
import logging
import spacy

logger = logging.getLogger(__name__)


def extract_animals(text, model_path):
    """Function to extract animals from text using ONLY the custom model."""
    nlp = spacy.load(model_path)

    # Process the text
    doc = nlp(text)

    # Select only entities labeled as "ANIMAL"
    found_animals = [ent.text.lower() for ent in doc.ents if ent.label_ == "ANIMAL"]

    return found_animals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str, help="Input text for NER")
    parser.add_argument('--model_path', type=str, default='custom_ner_model/')
    args = parser.parse_args()

    animals_found = extract_animals(args.text, args.model_path)
    logger.info(f"Found animals: {animals_found}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    main()
