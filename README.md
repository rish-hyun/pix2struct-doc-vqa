# pix2struct-docvqa-base

## Instructions

Following have been tested on Google Colab

- Create `config.toml` file with following content
    ```toml
    [MODEL]
    ORGANIZATION = "google"
    MODEL_NAME = "pix2struct-docvqa-base"
    MODELS_DIR = "models"
    ```

- Install the requirements (Better to run in a virtual environment!)
    ```bash
    pip install -r requirements.txt
    ```
- Download and convert HF model to ONNX with quantization
    ```bash
    python convert.py
    ```

- Run the inference

    Available Model Type:
    ```python
    available_models = {
        "HF_MODEL": Pix2StructHF,
        "ONNX_MODEL": Pix2StructOnnxWithoutPast,
        "ONNX_MODEL_WITH_PAST": Pix2StructOnnxWithPast,
    }
    ```

    ```bash
    python inference.py \
        --m <MODEL_TYPE> \
        --i <PATH_TO_IMAGE_FILE> \
        --q <QUESTION> \
        --quantize [True/False (Default: False)]
    ```

## Results

See benchmarking results [results.md](./results.md)