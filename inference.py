import os
import time
import toml
import argparse
import functools
import numpy as np

from PIL import Image
from onnxruntime import InferenceSession
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor


config = toml.load('config.toml')['MODEL']

MODEL_NAME = config['MODEL_NAME']
MODELS_DIR = config['MODELS_DIR']


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f">>> | {repr(func.__name__)} : {round(run_time, 3)} secs | <<<")
        return value
    return wrapper


class Pix2StructHF:

    def __init__(
        self,
        model_path: str = os.path.join(MODELS_DIR, MODEL_NAME),
        tokenizer_path: str = os.path.join(MODELS_DIR, MODEL_NAME),
    ) -> None:

        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.model = self._load_model(model_path)

    @timeit
    def _load_tokenizer(self, tokenizer_path):
        return Pix2StructProcessor.from_pretrained(tokenizer_path, local_files_only=True)

    @timeit
    def _preprocess(self, image_paths: list, questions: list):
        return self.tokenizer(
            images=[Image.open(_path) for _path in image_paths],
            text=questions,
            return_tensors="pt"
        )

    @timeit
    def _postprocess(self, outputs):
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    @timeit
    def _load_model(self, model_path):
        return Pix2StructForConditionalGeneration.from_pretrained(model_path, local_files_only=True)

    @timeit
    def _generate(self, inputs):
        self.model.eval()
        return self.model.generate(**inputs)

    def run(self, image_paths, questions):
        inputs = self._preprocess(image_paths, questions)
        outputs = self._generate(inputs)
        return self._postprocess(outputs)


class Pix2StructOnnxWithoutPast:

    def __init__(
        self,
        model_path=os.path.join(MODELS_DIR, MODEL_NAME) + '_onnx',
        tokenizer_path=os.path.join(MODELS_DIR, MODEL_NAME) + '_onnx',
        providers=['CPUExecutionProvider']
    ) -> None:

        self.providers = providers
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.encoder = self._load_encoder(model_path)
        self.decoder = self._load_decoder(model_path)

    @timeit
    def _load_tokenizer(self, tokenizer_path):
        return Pix2StructProcessor.from_pretrained(tokenizer_path, local_files_only=True)

    @timeit
    def _preprocess(self, image_paths: list, questions: list):
        inputs = self.tokenizer(
            images=[Image.open(_path) for _path in image_paths],
            text=questions,
            return_tensors="np"
        )
        inputs['attention_mask'] = inputs['attention_mask'].astype(np.int64)
        return inputs

    @timeit
    def _postprocess(self, outputs):
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    @timeit
    def _load_encoder(self, model_path):
        return InferenceSession(f'{model_path}/encoder_model.onnx', providers=self.providers)

    @timeit
    def _load_decoder(self, model_path):
        return InferenceSession(f'{model_path}/decoder_model.onnx', providers=self.providers)

    @timeit
    def _generate(self, inputs):
        last_hidden_state = self.encoder.run(['last_hidden_state'], {
            'flattened_patches': inputs['flattened_patches'],
            'attention_mask': inputs['attention_mask']
        })[0]

        batch_size = inputs['attention_mask'].shape[0]
        input_ids = np.zeros((batch_size, 1), dtype=np.int64)

        while (input_ids[:, -1] != 1).any():
            logits = self.decoder.run(['logits'], {
                'encoder_attention_mask': inputs['attention_mask'],
                'input_ids': input_ids,
                'decoder_attention_mask': np.ones_like(input_ids),
                'encoder_hidden_states': last_hidden_state
            })[0]

            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs /= np.sum(probs, axis=-1, keepdims=True)
            predicted_token = np.argmax(probs[:, -1, :], axis=-1)
            input_ids = np.concatenate(
                (input_ids, predicted_token[:, np.newaxis]),
                axis=-1
            )

        return input_ids

    def run(self, image_paths, questions):
        inputs = self._preprocess(image_paths, questions)
        outputs = self._generate(inputs)
        return self._postprocess(outputs)


class Pix2StructOnnxWithPast:

    def __init__(
        self,
        model_path=os.path.join(MODELS_DIR, MODEL_NAME) + '_onnx_with_past',
        tokenizer_path=os.path.join(
            MODELS_DIR, MODEL_NAME) + '_onnx_with_past',
        providers=['CPUExecutionProvider']
    ) -> None:

        self.providers = providers
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.encoder = self._load_encoder(model_path)
        self.decoder = self._load_decoder(model_path)
        self.decoder_with_past = self._load_decoder_with_past(model_path)

    @timeit
    def _load_tokenizer(self, tokenizer_path):
        return Pix2StructProcessor.from_pretrained(tokenizer_path, local_files_only=True)

    @timeit
    def _preprocess(self, image_paths: list, questions: list):
        inputs = self.tokenizer(
            images=[Image.open(_path) for _path in image_paths],
            text=questions,
            return_tensors="np"
        )
        inputs['attention_mask'] = inputs['attention_mask'].astype(np.int64)
        return inputs

    @timeit
    def _postprocess(self, outputs):
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    @timeit
    def _load_encoder(self, model_path):
        return InferenceSession(f'{model_path}/encoder_model.onnx', providers=self.providers)

    @timeit
    def _load_decoder(self, model_path):
        return InferenceSession(f'{model_path}/decoder_model.onnx', providers=self.providers)

    @timeit
    def _load_decoder_with_past(self, model_path):
        return InferenceSession(f'{model_path}/decoder_with_past_model.onnx', providers=self.providers)

    @timeit
    def _generate(self, inputs):
        last_hidden_state = self.encoder.run(['last_hidden_state'], {
            'flattened_patches': inputs['flattened_patches'],
            'attention_mask': inputs['attention_mask']
        })[0]

        batch_size = inputs['attention_mask'].shape[0]
        decoded_ids = input_ids = np.zeros((batch_size, 1), dtype=np.int64)

        decoder_inputs = {
            'encoder_attention_mask': inputs['attention_mask'],
            'input_ids': input_ids,
            'decoder_attention_mask': np.ones_like(input_ids),
            'encoder_hidden_states': last_hidden_state
        }

        logits, *values = self.decoder.run(None, decoder_inputs)

        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)
        predicted_token = np.argmax(probs, axis=-1)
        decoded_ids = np.concatenate((decoded_ids, predicted_token), axis=-1)

        decoder_inputs['input_ids'] = predicted_token
        for i in range(len(values)//4):
            decoder_inputs.update({
                f'past_key_values.{i}.decoder.key': values[4*i+0],
                f'past_key_values.{i}.decoder.value': values[4*i+1],
                f'past_key_values.{i}.encoder.key': values[4*i+2],
                f'past_key_values.{i}.encoder.value': values[4*i+3],
            })

        while (predicted_token[:, -1] != 1).any():  # Regressive Generation!

            logits, *values = self.decoder_with_past.run(None, decoder_inputs)

            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs /= np.sum(probs, axis=-1, keepdims=True)
            predicted_token = np.argmax(probs, axis=-1)
            decoded_ids = np.concatenate((decoded_ids, predicted_token), axis=-1)

            decoder_inputs['input_ids'] = predicted_token
            for i in range(len(values)//2):
                decoder_inputs.update({
                    f'past_key_values.{i}.decoder.key': values[2*i+0],
                    f'past_key_values.{i}.decoder.value': values[2*i+1],
                })

        return decoded_ids

    def run(self, image_paths, questions):
        inputs = self._preprocess(image_paths, questions)
        outputs = self._generate(inputs)
        return self._postprocess(outputs)


if __name__ == '__main__':

    available_models = {
        'HF_MODEL': Pix2StructHF,
        'ONNX_MODEL': Pix2StructOnnxWithoutPast,
        'ONNX_MODEL_WITH_PAST': Pix2StructOnnxWithPast,
    }

    parser = argparse.ArgumentParser(
        description="Run Pix2Struct Inference"
    )

    parser.add_argument(
        "--model", "-m",
        help="Path to the model folder",
        required=True,
        choices=available_models
    )

    parser.add_argument(
        "--images", "-i",
        help="Path to the images",
        required=True,
        nargs='+'
    )

    parser.add_argument(
        "--questions", "-q",
        help="Questions",
        required=True,
        nargs='+'
    )

    args = parser.parse_args()
    model = available_models.get(args.model)()
    answers = model.run(args.images, args.questions)

    print('<==============OUTPUT==============>')
    for q, a in zip(args.questions, answers):
        print(f'{q} -> {a}')
