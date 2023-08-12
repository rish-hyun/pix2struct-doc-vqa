# Benchmarking Results

---

Size of model `pix2struct-docvqa-base` : `1077` MB

Size of model `pix2struct-docvqa-large_onnx` : (351+727) MB = `1078` MB [Encoder + Decoder]

Size of model `pix2struct-docvqa-large_onnx_with_past` : (351+727+673) MB = `1751` MB [Encoder + Decoder + Decoder with Past]

---

Testing Sample
- Image: data/documents/xfbc0228_2.png
- Question: What is the total amount of the bill?
- Ground truth: ['March 2, 1951']
- Predicted answer: March 2, 1951

---

Best Time taken by each model on Testing sample [Ran on Google Colab CPU for 5 times]:

| Model                                  | Initialization Time (sec) | Inference Time (sec)|
| -------------------------------------- | ------------------------- | ------------------- |
| pix2struct-docvqa-base                 | `2.84`                    | `29.70`             |
| pix2struct-docvqa-large_onnx           | `2.14`                    | `28.37`             |
| pix2struct-docvqa-large_onnx_with_past | `3.79`                    | `17.5`              |

---

Conclusion:
- The model `pix2struct-docvqa-large_onnx_with_past` is the best model as it has the least inference time

---

Todo: Quantization of models can be done to reduce the size of the models.