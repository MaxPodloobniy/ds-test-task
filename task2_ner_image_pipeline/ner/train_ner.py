import spacy
import random
import os
import json
from spacy.training.example import Example


def load_data(filepath):
    """Function for loading json format dataset for model training and testing"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    data = []
    for item in data:
        entities = [(start, end, label) for start, end, label in item['entities']]
        data.append((item['text'], {"entities": entities}))

    return data


# Load the base model
nlp = spacy.load("en_core_web_sm")
CUSTOM_MODEL_PATH = "custom_ner_model"

# Add NER component if not already present
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Add custom label "ANIMAL"
ner.add_label("ANIMAL")

# Load training dataset
train_dataset = load_data("ner/data_for_ner/train.json")

# Check annotation consistency
for text, annotations in train_dataset:
    tags = spacy.training.offsets_to_biluo_tags(nlp.make_doc(text), annotations["entities"])
    if "-" in tags:
        print(f"❌ Annotation error: {text} -> {tags}")

# Disable other pipelines to prevent unnecessary retraining
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()

    # Training loop
    for i in range(10):
        random.shuffle(train_dataset)
        losses = {}
        for text, annotations in train_dataset:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.3, losses=losses)
        print(f"Iteration {i + 1}, Loss: {losses['ner']}")

    # Save the model
    print("Saving model...")
    os.makedirs(CUSTOM_MODEL_PATH, exist_ok=True)
    nlp.to_disk(CUSTOM_MODEL_PATH)
    print(f"Model saved to {os.path.abspath(CUSTOM_MODEL_PATH)}")

    # Verify the saved model
    try:
        nlp_loaded = spacy.load(CUSTOM_MODEL_PATH)
        print("Successfully loaded saved model")
    except Exception as e:
        print(f"Error loading saved model: {e}")


test_dataset = load_data("ner/data_for_ner/test.json")

for sentence in test_dataset:
    doc = nlp(sentence)
    print(f'\n{sentence}')

    for ent in doc.ents:
        print(f"Found: {ent.text} (category: {ent.label_})")
