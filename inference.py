import os
import toml
import argparse
import numpy as np

from PIL import Image
from onnxruntime import InferenceSession
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

from utils.time import TimeContextManager


config = toml.load("config.toml")["MODEL"]

MODEL_NAME = config["MODEL_NAME"]
MODELS_DIR = config["MODELS_DIR"]


class Processor:
    def __init__(self, processor_path: str) -> None:
        self.processor = self._load_processor(processor_path)

    def _load_processor(self, processor_path):
        return Pix2StructProcessor.from_pretrained(
            processor_path, local_files_only=True
        )

    def __call__(self, image_paths: list, questions: list, return_tensors: str):
        return self.processor(
            images=[Image.open(_path) for _path in image_paths],
            text=questions,
            return_tensors=return_tensors,
        )

    def decode(self, outputs):
        return self.processor.batch_decode(outputs, skip_special_tokens=True)


class Pix2StructHF:
    def __init__(
        self,
        model_path: str = os.path.join(MODELS_DIR, MODEL_NAME),
        quantize: bool = False,
    ) -> None:
        self.processor = Processor(model_path)
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        return Pix2StructForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True
        )

    def _generate(self, inputs):
        self.model.eval()
        return self.model.generate(**inputs)

    def run(self, image_paths, questions):
        inputs = self.processor(
            image_paths=image_paths, questions=questions, return_tensors="pt"
        )
        outputs = self._generate(inputs)
        return self.processor.decode(outputs)


class Pix2StructOnnxWithoutPast:
    def __init__(
        self,
        model_path=os.path.join(MODELS_DIR, MODEL_NAME) + "_onnx",
        quantize: bool = False,
        providers=["CPUExecutionProvider"],
    ) -> None:
        self.providers = providers
        self.processor = Processor(model_path)

        _models = self._load_model(model_path, quantize)
        self.encoder = _models.pop("encoder")
        self.decoder = _models.pop("decoder")

    def _load_encoder(self, model_path: str, quantize: bool):
        _path = f"{model_path}/encoder_model.onnx"
        if quantize:
            _path = _path.replace(".onnx", "_quantized.onnx")
        return InferenceSession(_path, providers=self.providers)

    def _load_decoder(self, model_path: str, quantize: bool):
        _path = f"{model_path}/decoder_model.onnx"
        if quantize:
            _path = _path.replace(".onnx", "_quantized.onnx")
        return InferenceSession(_path, providers=self.providers)

    def _load_model(self, model_path: str, quantize: bool):
        return {
            "encoder": self._load_encoder(model_path, quantize),
            "decoder": self._load_decoder(model_path, quantize),
        }

    def _generate(self, inputs):
        last_hidden_state = self.encoder.run(
            ["last_hidden_state"],
            {
                "flattened_patches": inputs["flattened_patches"],
                "attention_mask": inputs["attention_mask"],
            },
        )[0]

        batch_size = inputs["attention_mask"].shape[0]
        input_ids = np.zeros((batch_size, 1), dtype=np.int64)

        while (input_ids[:, -1] != 1).any():
            logits = self.decoder.run(
                ["logits"],
                {
                    "encoder_attention_mask": inputs["attention_mask"],
                    "input_ids": input_ids,
                    "decoder_attention_mask": np.ones_like(input_ids),
                    "encoder_hidden_states": last_hidden_state,
                },
            )[0]

            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs /= np.sum(probs, axis=-1, keepdims=True)
            predicted_token = np.argmax(probs[:, -1, :], axis=-1)
            input_ids = np.concatenate(
                (input_ids, predicted_token[:, np.newaxis]), axis=-1
            )

        return input_ids

    def run(self, image_paths, questions):
        inputs = self.processor(
            image_paths=image_paths, questions=questions, return_tensors="np"
        )
        inputs["attention_mask"] = inputs["attention_mask"].astype(np.int64)

        outputs = self._generate(inputs)
        return self.processor.decode(outputs)


class Pix2StructOnnxWithPast:
    def __init__(
        self,
        model_path=os.path.join(MODELS_DIR, MODEL_NAME) + "_onnx_with_past",
        quantize: bool = False,
        providers=["CPUExecutionProvider"],
    ) -> None:
        self.providers = providers
        self.processor = Processor(model_path)

        _models = self._load_model(model_path, quantize)
        self.encoder = _models.pop("encoder")
        self.decoder = _models.pop("decoder")
        self.decoder_with_past = _models.pop("decoder_with_past")

    def _load_encoder(self, model_path: str, quantize: bool):
        _path = f"{model_path}/encoder_model.onnx"
        if quantize:
            _path = _path.replace(".onnx", "_quantized.onnx")
        return InferenceSession(_path, providers=self.providers)

    def _load_decoder(self, model_path: str, quantize: bool):
        _path = f"{model_path}/decoder_model.onnx"
        if quantize:
            _path = _path.replace(".onnx", "_quantized.onnx")
        return InferenceSession(_path, providers=self.providers)

    def _load_decoder_with_past(self, model_path: str, quantize: bool):
        _path = f"{model_path}/decoder_with_past_model.onnx"
        if quantize:
            _path = _path.replace(".onnx", "_quantized.onnx")
        return InferenceSession(_path, providers=self.providers)

    def _load_model(self, model_path: str, quantize: bool):
        return {
            "encoder": self._load_encoder(model_path, quantize),
            "decoder": self._load_decoder(model_path, quantize),
            "decoder_with_past": self._load_decoder_with_past(model_path, quantize),
        }

    def _generate(self, inputs):
        last_hidden_state = self.encoder.run(
            ["last_hidden_state"],
            {
                "flattened_patches": inputs["flattened_patches"],
                "attention_mask": inputs["attention_mask"],
            },
        )[0]

        batch_size = inputs["attention_mask"].shape[0]
        decoded_ids = input_ids = np.zeros((batch_size, 1), dtype=np.int64)

        decoder_inputs = {
            "encoder_attention_mask": inputs["attention_mask"],
            "input_ids": input_ids,
            "decoder_attention_mask": np.ones_like(input_ids),
            "encoder_hidden_states": last_hidden_state,
        }

        logits, *values = self.decoder.run(None, decoder_inputs)

        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)
        predicted_token = np.argmax(probs, axis=-1)
        decoded_ids = np.concatenate((decoded_ids, predicted_token), axis=-1)

        decoder_inputs["input_ids"] = predicted_token
        for i in range(len(values) // 4):
            decoder_inputs.update(
                {
                    f"past_key_values.{i}.decoder.key": values[4 * i + 0],
                    f"past_key_values.{i}.decoder.value": values[4 * i + 1],
                    f"past_key_values.{i}.encoder.key": values[4 * i + 2],
                    f"past_key_values.{i}.encoder.value": values[4 * i + 3],
                }
            )

        while (predicted_token[:, -1] != 1).any():  # Regressive Generation!
            logits, *values = self.decoder_with_past.run(None, decoder_inputs)

            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs /= np.sum(probs, axis=-1, keepdims=True)
            predicted_token = np.argmax(probs, axis=-1)
            decoded_ids = np.concatenate((decoded_ids, predicted_token), axis=-1)

            decoder_inputs["input_ids"] = predicted_token
            for i in range(len(values) // 2):
                decoder_inputs.update(
                    {
                        f"past_key_values.{i}.decoder.key": values[2 * i + 0],
                        f"past_key_values.{i}.decoder.value": values[2 * i + 1],
                    }
                )

        return decoded_ids

    def run(self, image_paths, questions):
        inputs = self.processor(
            image_paths=image_paths, questions=questions, return_tensors="np"
        )
        inputs["attention_mask"] = inputs["attention_mask"].astype(np.int64)

        outputs = self._generate(inputs)
        return self.processor.decode(outputs)


available_models = {
    "HF_MODEL": Pix2StructHF,
    "ONNX_MODEL": Pix2StructOnnxWithoutPast,
    "ONNX_MODEL_WITH_PAST": Pix2StructOnnxWithPast,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pix2Struct Inference")

    parser.add_argument(
        "--model",
        "-m",
        help="Path to the model folder",
        required=True,
        choices=available_models,
    )

    parser.add_argument(
        "--quantize",
        help="Quantize the model",
        default=False,
    )

    parser.add_argument(
        "--images", "-i", help="Path to the images", required=True, nargs="+"
    )

    parser.add_argument("--questions", "-q", help="Questions", required=True, nargs="+")

    args = parser.parse_args()

    tcm = TimeContextManager()

    cls = available_models.get(args.model)
    model = tcm("model_init_time").measure(cls)(quantize=args.quantize)
    answers = tcm("model_run_time").measure(model.run)(args.images, args.questions)

    print("<==============OUTPUT==============>")
    for q, a in zip(args.questions, answers):
        print(f"Question: {q} \nAnswer: {a} \n")
