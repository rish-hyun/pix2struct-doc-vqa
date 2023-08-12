import os
import toml


config = toml.load("config.toml")["MODEL"]

ORGANIZATION = config["ORGANIZATION"]
MODEL_NAME = config["MODEL_NAME"]
MODELS_DIR = config["MODELS_DIR"]

HF_MODEL_DIR = os.path.join(MODELS_DIR, MODEL_NAME)

if not os.path.isfile(os.path.join(HF_MODEL_DIR, "pytorch_model.bin")):
    command = (
        f"git clone https://huggingface.co/{ORGANIZATION}/{MODEL_NAME} {HF_MODEL_DIR}"
    )
    os.system(command)


def convert_to_onnx(with_past: bool):
    task = "visual-question-answering"
    output = f"{HF_MODEL_DIR}_onnx"

    if with_past:
        task += "-with-past"
        output += "_with_past"

    command = f"optimum-cli export onnx --model {HF_MODEL_DIR} --task {task} {output}"
    os.system(command)


if __name__ == "__main__":
    convert_to_onnx(with_past=False)
    convert_to_onnx(with_past=True)
