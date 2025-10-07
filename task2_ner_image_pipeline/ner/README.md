# Named Entity Recognition (NER) Model

This directory contains a custom Named Entity Recognition model built with spaCy for extracting
animal names from text. The model was trained to recognize a specific "ANIMAL" entity type and
can identify various animal mentions in natural language sentences.

## Model Overview

The NER model is based on spaCy's `en_core_web_md` base model with a custom entity type added
to the pipeline. During training, the "ANIMAL" entity was added to the NER component, allowing
the model to specifically recognize and extract animal names from text with high accuracy.

## Dataset

The dataset was manually created through a two-step process. First, a Large Language Model was
used to generate diverse sentences containing target animals in various contexts. Then, the entity
indices were manually annotated to mark the exact positions where animal names appear in each
sentence.

The dataset is split into training and test sets, both stored in JSON format with the following
structure:

```json
[
    {
        "text": "I saw a chimpanzee at the zoo yesterday.",
        "entities": [[8, 18, "ANIMAL"]]
    }
]
```

Each entry contains the text and a list of entities with their start index, end index, and label.

## Training Configuration

The model was trained using the following parameters and achieved good performance on the
test set:

**Training Results:**
- Precision: 92.86%
- Recall: 97.50%
- F1-Score: 95.12%

These metrics demonstrate that the model can reliably identify animal entities in text with minimal
false positives and very few missed detections.

## Usage

### Training the Model

The training script can be executed from the terminal with customizable parameters. To train the
model, use:

```bash
python train_ner.py \
    --epochs 5 \
    --dropout 0.3 \
    --model_path custom_ner_model/ \
    --train_data data_for_ner/train.json \
    --test_data data_for_ner/test.json
```

**Available Arguments:**
- `--epochs` (default: 5) - Number of training iterations over the dataset
- `--dropout` (default: 0.3) - Dropout rate for regularization during training
- `--model_path` (default: custom_ner_model/) - Directory where the trained model will be saved
- `--train_data` (default: data_for_ner/train.json) - Path to the training dataset
- `--test_data` (default: data_for_ner/test.json) - Path to the test dataset

The training process will display loss values for each epoch and automatically evaluate the model on the test set after training completes. The final model is saved to the specified directory and can be loaded for inference.

### Making Predictions

To extract animal entities from text, use the inference script. It accepts a text string and returns all detected animal names:

```bash
python infer_ner.py "I saw a chimpanzee and a coyote in the forest." \
    --model_path custom_ner_model/
```

**Available Arguments:**
- `text` (positional, required) - The input text to analyze for animal entities
- `--model_path` (default: custom_ner_model/) - Path to the trained NER model directory

**Example Usage:**

```bash
# Simple sentence
python infer_ner.py "The chimpanzee was eating bananas."

# Multiple animals
python infer_ner.py "I encountered a coyote and saw a chimpanzee at the zoo."

# Complex sentence
python infer_ner.py "While hiking through the forest yesterday, I spotted a coyote near the trail."
```

The script outputs the detected animals in lowercase format as a list, making it easy to integrate with other components of the pipeline.

## Model Behavior

The model extracts only entities labeled as "ANIMAL" and ignores all other entity types that might be present in the base spaCy model. All extracted animal names are converted to lowercase for consistency, which simplifies comparison and matching operations in downstream tasks.

The model handles various linguistic contexts including:
- Simple declarative sentences with single animal mentions
- Complex sentences with multiple clauses and animal references
- Different grammatical positions (subjects, objects, prepositional phrases)
- Various verb tenses and sentence structures
- Capitalized and lowercase animal names

## Integration

This NER model is designed to work as part of a larger pipeline that combines image classification with text analysis. The extracted animal names can be compared against image classification results to verify whether statements about animals match corresponding visual content.
