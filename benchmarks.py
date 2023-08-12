import os
import json
from inference import available_models, timeit


DATA_DIR = "data"
QA_FILE = "val_v1.0.json"
TEST_BATCHES = [1, 3, 5, 7, 10]

with open(os.path.join(DATA_DIR, QA_FILE), "r") as f:
    images, questions, answers = zip(
        *[(item["image"], item["question"], item["answers"]) for item in json.load(f)]
    )

for i in TEST_BATCHES:
    for model_type, cls in available_models.items():
        model = cls()
        predicted_answers = timeit(model.run(images[:i], questions[:i]))

        print(f"<====== OUTPUT for Test Batch Size: {i}, Model: {model_type} ========>")
        for q, p, a in zip(questions[:i], predicted_answers[:i], answers[:i]):
            print("Question:", q)
            print("Predicted answer:", p)
            print("Ground truth:", a)
            print()
