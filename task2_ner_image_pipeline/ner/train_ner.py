import spacy
import random
import os
import json
import argparse
import subprocess
import sys
from spacy.training.example import Example
from spacy.scorer import Scorer
from spacy.util import minibatch, compounding


def parse_args():
    """Function for parsing arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--model_path', type=str, default='custom_ner_model/')
    parser.add_argument('--train_data', type=str, default='data_for_ner/train.json')
    parser.add_argument('--test_data', type=str, default='data_for_ner/test.json')
    return parser.parse_args()


def load_data(filepath):
    """Function for loading json format dataset for model training and testing"""
    with open(filepath, 'r') as f:
        loaded_data = json.load(f)

    data = []
    for item in loaded_data:
        entities = [(start, end, label) for start, end, label in item['entities']]
        data.append((item['text'], {"entities": entities}))

    return data

def load_or_download_model(model_name):
    """Load spaCy model, download if not available"""
    try:
        nlp = spacy.load(model_name)
        print(f"Model '{model_name}' loaded")
        return nlp
    except OSError:
        print(f"Downloading model '{model_name}'...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        nlp = spacy.load(model_name)
        print(f"Model '{model_name}' downloaded and loaded")
        return nlp

def main():
    # Loading all arguments
    args = parse_args()

    # Load the base model
    nlp = load_or_download_model("en_core_web_md")

    # Add NER component if not already present
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Add custom label "ANIMAL"
    ner.add_label("ANIMAL")

    # Load training dataset
    train_dataset = load_data(args.train_data)
    test_dataset = load_data(args.test_data)

    print("Train dataset length:", len(train_dataset))
    print("Test dataset length:", len(test_dataset))

    # Prepare training examples for initialization
    train_examples = []
    for text, annotations in train_dataset:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        train_examples.append(example)

    # Disable other pipelines to prevent unnecessary retraining
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.initialize(get_examples=lambda: train_examples)

        # Training loop
        for i in range(args.epochs):
            random.shuffle(train_examples)
            losses = {}

            # Update in batches for better stability
            batches = minibatch(train_examples, size=compounding(2.0, 8.0, 1.001))
            for batch in batches:
                nlp.update(batch, drop=args.dropout, losses=losses, sgd=optimizer)

            print(f"Iteration {i + 1}, Loss: {losses['ner']}")

        # Save the model
        print("Saving model...")
        os.makedirs(args.model_path, exist_ok=True)
        nlp.to_disk(args.model_path)
        print(f"Model saved to {os.path.abspath(args.model_path)}")

        # Verify the saved model
        try:
            nlp_loaded = spacy.load(args.model_path)
            print("Successfully loaded saved model")
        except Exception as e:
            print(f"Error loading saved model: {e}")

    # Testing saved model
    print("\n📊 Evaluating on test data...")
    examples = []
    for text, annotations in test_dataset:
        doc = nlp(text)

        example = Example.from_dict(doc, annotations)
        examples.append(example)

    scorer = Scorer()
    scores = scorer.score(examples)

    print(f"Precision: {scores['ents_p']:.2%}")
    print(f"Recall: {scores['ents_r']:.2%}")
    print(f"F1-Score: {scores['ents_f']:.2%}")


if __name__ == '__main__':
    main()
