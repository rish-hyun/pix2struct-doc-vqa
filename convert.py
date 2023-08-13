import os
import toml
from glob import glob
from onnxruntime.quantization import quantize_dynamic, QuantType


config = toml.load("config.toml")["MODEL"]

ORGANIZATION = config["ORGANIZATION"]
MODEL_NAME = config["MODEL_NAME"]
MODELS_DIR = config["MODELS_DIR"]

HF_MODEL_DIR = os.path.join(MODELS_DIR, MODEL_NAME)

if not os.path.isfile(os.path.join(HF_MODEL_DIR, "pytorch_model.bin")):
    os.system(
        f"git clone https://huggingface.co/{ORGANIZATION}/{MODEL_NAME} {HF_MODEL_DIR}"
    )


def convert_to_onnx(with_past: bool):
    task = "visual-question-answering"
    output = f"{HF_MODEL_DIR}_onnx"

    if with_past:
        task += "-with-past"
        output += "_with_past"

    command = f"optimum-cli export onnx --model {HF_MODEL_DIR} --task {task} {output}"
    os.system(command)
    return output


def quantize_onnx(onnx_model_path: str):
    for onnx_model in glob(os.path.join(onnx_model_path, "*.onnx")):
        quantized_model = onnx_model.replace(".onnx", "_quantized.onnx")
        quantize_dynamic(onnx_model, quantized_model, weight_type=QuantType.QInt8)


if __name__ == "__main__":
    _ = convert_to_onnx(with_past=False)
    model_path = convert_to_onnx(with_past=True)

    quantize_onnx(model_path)
