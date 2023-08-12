import os
import json
from inference import available_models

DATA_DIR = "data"

with open(os.path.join(DATA_DIR, "val_v1.0.json"), "r") as f:
    images, questions, answers = zip(
        *[(item["image"], item["question"], item["answers"]) for item in json.load(f)]
    )
