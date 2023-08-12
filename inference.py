import time
import functools

from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor


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

    def __init__(self, model_path, tokenizer_path, device="cpu"):
        self.device = device
        self.model = self._load_model(model_path)
        self.tokenizer = self._load_tokenizer(tokenizer_path)

    @timeit
    def _load_model(self, model_path):
        return Pix2StructForConditionalGeneration.from_pretrained(model_path, local_files_only=True).to(self.device)

    @timeit
    def _load_tokenizer(self, tokenizer_path):
        return Pix2StructProcessor.from_pretrained(tokenizer_path, local_files_only=True)

    @timeit
    def _preprocess(self, image_path, question):
        return self.tokenizer(
            images=Image.open(image_path),
            text=question,
            return_tensors="pt"
        ).to(self.device)

    @timeit
    def _postprocess(self, outputs):
        return self.tokenizer.decode(outputs, skip_special_tokens=True)

    @timeit
    def _generate(self, inputs):
        self.model.eval()
        return self.model.generate(**inputs)

    def run(self, image_path, question):
        self.model.eval()
        inputs = self._preprocess(image_path, question)
        outputs = self._generate(inputs)
        return self._postprocess(outputs)


class Pix2StructOnnxWithoutPast:
    ...


class Pix2StructOnnxWithPast:
    ...
