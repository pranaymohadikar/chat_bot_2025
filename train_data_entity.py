import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import random
import json

# Load TRAIN_DATA
with open("train_data_entity.json", "r") as file:
    TRAIN_DATA = json.load(file)

# Load a pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

# Add entity labels to the pipeline
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Disable other pipeline components during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.create_optimizer()
    for itn in range(20):  # Train for 10 iterations
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.5, losses=losses)
        print(f"Iteration {itn}, Losses: {losses}")

# Save the trained model
nlp.to_disk("custom_entity_model")