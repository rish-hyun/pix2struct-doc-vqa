import os
import toml
import json

from inference import available_models
from utils.time import TimeContextManager


config = toml.load("config.toml")["BENCHMARK"]

DATA_DIR = config["DATA_DIR"]
QA_FILE = config["QA_FILE"]

if __name__ == "__main__":
    images, questions, answers = [], [], []
    with open(os.path.join(DATA_DIR, QA_FILE), "r") as f:
        for item in json.load(f):
            images.append(os.path.join(DATA_DIR, item["image"]))
            questions.append(item["question"])
            answers.append(item["answers"])

    model_benchmark = {}
    for model_type, cls in available_models.items():
        for quantize in [True, False]:
            tcm = TimeContextManager(verbose=False)
            n = 1  # Number of samples to test

            for _ in range(5):
                model = tcm("model_init_time").measure(cls)(quantize=quantize)
                predicted_answers = tcm("model_run_time").measure(model.run)(
                    images[:n], questions[:n]
                )

            model_benchmark[
                model_type + "_quantized" if quantize else model_type
            ] = tcm.manager
            json.dump(model_benchmark, open("model_benchmarks.json", "w"), indent=4)

            print(
                f"<= OUTPUT for Test Batch Size: {n}, Model: {model_type}, Quantize: {quantize} =>"
            )

            for k in range(n):
                print("Question:", questions[k])
                print("Predicted answer:", predicted_answers[k])
                print("Ground truth:", answers[k])
                print()
