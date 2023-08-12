import os
import toml
import json
from inference import available_models


config = toml.load("config.toml")["BENCHMARK"]

DATA_DIR = config["DATA_DIR"]
QA_FILE = config["QA_FILE"]
TEST_BATCHES = config["TEST_BATCHES"]

images, questions, answers = [], [], []
with open(os.path.join(DATA_DIR, QA_FILE), "r") as f:
    for item in json.load(f):
        images.append(os.path.join(DATA_DIR, item["image"]))
        questions.append(item["question"])
        answers.append(item["answers"])

for index in TEST_BATCHES:
    for model_type, cls in available_models.items():
        model = cls()
        predicted_answers = model.run(images[:index], questions[:index])

        print(f"<==== OUTPUT for Test Batch Size: {index}, Model: {model_type} =====>")
        for k in range(index):
            print("Question:", questions[k])
            print("Predicted answer:", predicted_answers[k])
            print("Ground truth:", answers[k])
            print()
