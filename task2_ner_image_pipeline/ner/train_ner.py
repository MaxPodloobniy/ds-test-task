import spacy
import random
import os
from spacy.training.example import Example

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

TRAIN_DATA = [
    ("I saw a chimpanzee climbing a tree.", {"entities": [(8, 18, "ANIMAL")]}),
    ("The chimpanzee was eating bananas.", {"entities": [(4, 14, "ANIMAL")]}),
    ("Chimpanzees are intelligent animals.", {"entities": [(0, 11, "ANIMAL")]}),
    ("A group of chimpanzees was making loud noises.", {"entities": [(11, 22, "ANIMAL")]}),
    ("I watched a documentary about chimpanzees.", {"entities": [(30, 41, "ANIMAL")]}),
    ("A chimpanzee can use tools to get food.", {"entities": [(2, 12, "ANIMAL")]}),
    ("We watched a playful chimpanzee at the zoo.", {"entities": [(21, 31, "ANIMAL")]}),
    ("Chimpanzees are intelligent animals.", {"entities": [(0, 11, "ANIMAL")]}),

    ("A coyote ran across the road.", {"entities": [(2, 8, "ANIMAL")]}),
    ("I heard a coyote howling at night.", {"entities": [(10, 16, "ANIMAL")]}),
    ("Coyotes live in the wild and hunt for food.", {"entities": [(0, 7, "ANIMAL")]}),
    ("The coyote was looking for food.", {"entities": [(4, 10, "ANIMAL")]}),
    ("Have you ever seen a coyote in real life?", {"entities": [(21, 27, "ANIMAL")]}),
    ("A coyote was hunting in the desert.", {"entities": [(2, 8, "ANIMAL")]}),
    ("I heard a coyote howling at night.", {"entities": [(10, 16, "ANIMAL")]}),
    ("Coyotes are found in North America.", {"entities": [(0, 7, "ANIMAL")]}),
    ("A coyote ran across the road.", {"entities": [(2, 8, "ANIMAL")]}),
    ("Coyotes adapt well to urban areas.", {"entities": [(0, 7, "ANIMAL")]}),

    ("A deer jumped over the fence.", {"entities": [(2, 6, "ANIMAL")]}),
    ("The deer was grazing in the field.", {"entities": [(4, 8, "ANIMAL")]}),
    ("I spotted a group of deer in the forest.", {"entities": [(21, 25, "ANIMAL")]}),
    ("Deer are common in this area.", {"entities": [(0, 4, "ANIMAL")]}),
    ("A baby deer is called a fawn.", {"entities": [(7, 11, "ANIMAL")]}),
    ("The deer was grazing near the river.", {"entities": [(4, 8, "ANIMAL")]}),
    ("I saw a deer in the forest.", {"entities": [(8, 12, "ANIMAL")]}),
    ("Deer are common in this region.", {"entities": [(0, 4, "ANIMAL")]}),
    ("A deer crossed the road suddenly.", {"entities": [(2, 6, "ANIMAL")]}),
    ("The deer had large antlers.", {"entities": [(4, 8, "ANIMAL")]}),

    ("I saw a duck swimming in the lake.", {"entities": [(8, 12, "ANIMAL")]}),
    ("The duck quacked loudly.", {"entities": [(4, 8, "ANIMAL")]}),
    ("Ducks are often found in ponds.", {"entities": [(0, 5, "ANIMAL")]}),
    ("A mother duck protects her ducklings.", {"entities": [(9, 13, "ANIMAL")]}),
    ("Have you ever fed ducks at the park?", {"entities": [(18, 23, "ANIMAL")]}),
    ("A duck was swimming in the pond.", {"entities": [(2, 6, "ANIMAL")]}),
    ("The duck quacked loudly.", {"entities": [(4, 8, "ANIMAL")]}),
    ("I saw a duck at the lake.", {"entities": [(8, 12, "ANIMAL")]}),
    ("Ducks migrate during the winter.", {"entities": [(0, 5, "ANIMAL")]}),
    ("A mother duck led her ducklings.", {"entities": [(9, 13, "ANIMAL")]}),

    ("An eagle soared high in the sky.", {"entities": [(3, 8, "ANIMAL")]}),
    ("The eagle caught a fish.", {"entities": [(4, 9, "ANIMAL")]}),
    ("Eagles have excellent vision.", {"entities": [(0, 6, "ANIMAL")]}),
    ("I saw a golden eagle in the mountains.", {"entities": [(15, 20, "ANIMAL")]}),
    ("Eagles build nests on tall cliffs.", {"entities": [(0, 6, "ANIMAL")]}),
    ("The eagle soared above the mountains.", {"entities": [(4, 9, "ANIMAL")]}),
    ("I saw an eagle catching a fish.", {"entities": [(9, 14, "ANIMAL")]}),
    ("Eagles have excellent vision.", {"entities": [(0, 6, "ANIMAL")]}),
    ("An eagle built a nest on the cliff.", {"entities": [(3, 8, "ANIMAL")]}),
    ("The eagle spread its wings wide.", {"entities": [(4, 9, "ANIMAL")]}),

    ("An elephant was drinking water.", {"entities": [(3, 11, "ANIMAL")]}),
    ("Elephants are the largest land animals.", {"entities": [(0, 9, "ANIMAL")]}),
    ("I saw a herd of elephants in the zoo.", {"entities": [(16, 25, "ANIMAL")]}),
    ("The elephant used its trunk to grab food.", {"entities": [(4, 12, "ANIMAL")]}),
    ("Elephants have strong memories.", {"entities": [(0, 9, "ANIMAL")]}),
    ("The elephant sprayed water with its trunk.", {"entities": [(4, 12, "ANIMAL")]}),
    ("I saw a baby elephant at the zoo.", {"entities": [(13, 21, "ANIMAL")]}),
    ("Elephants have strong memories.", {"entities": [(0, 9, "ANIMAL")]}),
    ("An elephant walked through the jungle.", {"entities": [(3, 11, "ANIMAL")]}),
    ("The elephant flapped its ears.", {"entities": [(4, 12, "ANIMAL")]}),

    ("A hedgehog curled into a ball.", {"entities": [(2, 10, "ANIMAL")]}),
    ("I saw a hedgehog in my backyard.", {"entities": [(8, 16, "ANIMAL")]}),
    ("Hedgehogs have spiky fur.", {"entities": [(0, 9, "ANIMAL")]}),
    ("The hedgehog was hiding under the leaves.", {"entities": [(4, 12, "ANIMAL")]}),
    ("Hedgehogs are nocturnal animals.", {"entities": [(0, 9, "ANIMAL")]}),
    ("A hedgehog rolled into a ball.", {"entities": [(2, 10, "ANIMAL")]}),
    ("The hedgehog has sharp spines.", {"entities": [(4, 12, "ANIMAL")]}),
    ("Hedgehogs are nocturnal animals.", {"entities": [(0, 9, "ANIMAL")]}),
    ("I saw a tiny hedgehog in the garden.", {"entities": [(13, 21, "ANIMAL")]}),
    ("The hedgehog was hiding under the leaves.", {"entities": [(4, 12, "ANIMAL")]}),

    ("A hippopotamus was resting in the water.", {"entities": [(2, 14, "ANIMAL")]}),
    ("Hippopotamuses spend most of their time in water.", {"entities": [(0, 14, "ANIMAL")]}),
    ("I saw a baby hippopotamus at the zoo.", {"entities": [(13, 25, "ANIMAL")]}),
    ("Hippopotamuses have powerful jaws.", {"entities": [(0, 14, "ANIMAL")]}),
    ("A hippopotamus can run surprisingly fast.", {"entities": [(2, 14, "ANIMAL")]}),
    ("A hippopotamus was bathing in the river.", {"entities": [(2, 14, "ANIMAL")]}),
    ("The hippopotamus is a large mammal.", {"entities": [(4, 16, "ANIMAL")]}),
    ("The hippopotamus yawned widely.", {"entities": [(4, 16, "ANIMAL")]}),

    ("A kangaroo hopped across the field.", {"entities": [(2, 10, "ANIMAL")]}),
    ("Kangaroos carry their babies in pouches.", {"entities": [(0, 9, "ANIMAL")]}),
    ("I saw a kangaroo at the wildlife park.", {"entities": [(8, 16, "ANIMAL")]}),
    ("The kangaroo was eating grass.", {"entities": [(4, 12, "ANIMAL")]}),
    ("Kangaroos are native to Australia.", {"entities": [(0, 9, "ANIMAL")]}),
    ("The kangaroo carried a baby in its pouch.", {"entities": [(4, 12, "ANIMAL")]}),
    ("Kangaroos are strong jumpers.", {"entities": [(0, 9, "ANIMAL")]}),
    ("I saw a kangaroo boxing with another one.", {"entities": [(8, 16, "ANIMAL")]}),
    ("The kangaroo was resting in the shade.", {"entities": [(4, 12, "ANIMAL")]}),

    ("A rhinoceros was drinking water.", {"entities": [(2, 12, "ANIMAL")]}),
    ("Rhinoceroses have thick skin.", {"entities": [(0, 12, "ANIMAL")]}),
    ("I saw a white rhinoceros at the zoo.", {"entities": [(14, 24, "ANIMAL")]}),
    ("Rhinoceroses use their horns for defense.", {"entities": [(0, 12, "ANIMAL")]}),
    ("A baby rhinoceros stays close to its mother.", {"entities": [(7, 17, "ANIMAL")]}),
    ("A rhinoceros has a thick skin.", {"entities": [(2, 12, "ANIMAL")]}),
    ("The rhinoceros was drinking water.", {"entities": [(4, 14, "ANIMAL")]}),
    ("Rhinoceroses have a strong horn.", {"entities": [(0, 12, "ANIMAL")]}),
    ("I saw a huge rhinoceros at the safari.", {"entities": [(13, 23, "ANIMAL")]}),
    ("The rhinoceros charged at the jeep.", {"entities": [(4, 14, "ANIMAL")]}),

    ("A tiger was stalking its prey.", {"entities": [(2, 7, "ANIMAL")]}),
    ("Tigers are powerful hunters.", {"entities": [(0, 6, "ANIMAL")]}),
    ("I saw a Bengal tiger at the zoo.", {"entities": [(15, 20, "ANIMAL")]}),
    ("The tiger roared loudly.", {"entities": [(4, 9, "ANIMAL")]}),
    ("Tigers have distinctive stripes.", {"entities": [(0, 6, "ANIMAL")]}),
    ("A tiger was stalking its prey.", {"entities": [(2, 7, "ANIMAL")]}),
    ("The tiger roared loudly in the jungle.", {"entities": [(4, 9, "ANIMAL")]}),
    ("Tigers are excellent hunters.", {"entities": [(0, 6, "ANIMAL")]}),
    ("I saw a white tiger at the zoo.", {"entities": [(14, 19, "ANIMAL")]}),
    ("The tiger pounced on its prey.", {"entities": [(4, 9, "ANIMAL")]}),
]

# Check annotation consistency
for text, annotations in TRAIN_DATA:
    tags = spacy.training.offsets_to_biluo_tags(nlp.make_doc(text), annotations["entities"])
    if "-" in tags:
        print(f"❌ Annotation error: {text} -> {tags}")

# Disable other pipelines to prevent unnecessary retraining
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()

    # Training loop
    for i in range(10):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
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

TEST_DATA = [
    ("There is a chimpanzee in this image."),
    ("I can see a chimpanzee in the photo."),
    ("This picture contains a chimpanzee."),

    ("There is a coyote in this picture."),
    ("I found a coyote in this image."),
    ("This photo clearly shows a coyote."),

    ("A deer is visible in this picture."),
    ("I can spot a deer in the photo."),
    ("This image includes a deer."),

    ("There is a duck in the picture."),
    ("You can see a duck in this image."),
    ("A duck is present in this photograph."),

    ("An eagle appears in this picture."),
    ("This photo has an eagle in it."),
    ("There is an eagle in the image."),

    ("You can see an elephant in this picture."),
    ("An elephant is present in the photo."),
    ("There is an elephant in this image."),

    ("A hedgehog is visible in this picture."),
    ("This photo has a hedgehog in it."),
    ("I can see a hedgehog in this image."),

    ("There is a hippopotamus in this picture."),
    ("A hippopotamus is clearly visible in this image."),
    ("This photo contains a hippopotamus."),

    ("You can find a kangaroo in this picture."),
    ("This image includes a kangaroo."),
    ("A kangaroo appears in this photo."),

    ("There is a rhinoceros in this image."),
    ("A rhinoceros is clearly visible in this picture."),
    ("This photo contains a rhinoceros."),

    ("I see a tiger in this image."),
    ("This picture has a tiger in it."),
    ("A tiger is present in this photo."),
]

for sentence in TEST_DATA:
    doc = nlp(sentence)
    print(f'\n{sentence}')

    for ent in doc.ents:
        print(f"Found: {ent.text} (category: {ent.label_})")
