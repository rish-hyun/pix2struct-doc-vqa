## Task: 
Convert Pix2Struct-docvqa-base Model to ONNX

---

## Objective:

The objective of this task is to take the Pix2Struct base model from the Hugging Face library (https://huggingface.co/google/pix2struct-docvqa-base) and convert it into the ONNX format, benchmarking results, comparing the inference time and model size of the original Hugging Face model with its ONNX equivalent.

---

## Relevant Theory:

- https://huggingface.co/google/pix2struct-docvqa-base
- https://huggingface.co/blog/convert-transformers-to-onnx
- https://huggingface.co/docs/transformers/main/en/serialization
- https://huggingface.co/docs/transformers/main/model_doc/pix2struct
- https://github.com/huggingface/notebooks/blob/main/examples/onnx-export.ipynb
- https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_pix2struct.ipynb

<br>

- https://github.com/google-research/pix2struct#docvqa
- https://rrc.cvc.uab.es/?ch=17&com=downloads

<br>

- https://www.analyticsvidhya.com/blog/2023/04/document-information-extraction-using-pix2struct/
- https://github.com/khadkechetan/information_extraction/blob/main/Docvqa/pix2struct/Pix2Struct_docVQA.ipynb

<br>

- https://arxiv.org/pdf/2210.03347.pdf
- https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=1

---

## Instructions:

### Model Conversion (convert.py):
A Python script, `convert.py`, which is used to convert the Hugging Face model into the ONNX format ensuring that the ONNX conversion process maintains the integrity and accuracy of the original model.

### Inference Script (inference.py):
Another Python script, `inference.py`, that is used to run the ONNX model. The script takes the path to the ONNX model folder and the image path as an argument and it is runnable from the command line as follows:

```
python inference.py -m <ONNX_model_folder_path> -i <IMAGE PATH>
```

This script is capable of loading the ONNX model and performing inference on a sample input image and then print the result in the terminal.

### Model Folder (model):
A folder named `model` that contains the ONNX converted files generated from the `convert.py script`.

### Benchmarking (results.md):
A Markdown file named `results.md` that document the benchmarking results. Conduct benchmarking on the original Hugging Face model and its ONNX counterpart.

Measure and record the following:
- Size of the Hugging Face model (in MB).
- Size of the ONNX model (in MB).
- Inference time of the Hugging Face model for a single example image (provide the time in seconds).
- Inference time of the ONNX model for the same example image (provide the time in seconds).
- Any other information or instruction

---
