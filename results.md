# Benchmarking Results

---

| Model                                             | Size                         | Files                                   |
| ------------------------------------------------- | ---------------------------- | --------------------------------------- |
| pix2struct-docvqa-base                            | `1077` MB                    | `pytorch_model.bin`                     |
| pix2struct-docvqa-base_onnx                       | (351+727) MB = `1078` MB     | `Encoder + Decoder`                     |
| pix2struct-docvqa-base_onnx (Quantized)           | (88+183) MB = `271` MB       | `Encoder + Decoder`                     |
| pix2struct-docvqa-base_onnx_with_past             | (351+727+673) MB = `1751` MB | `Encoder + Decoder + Decoder with Past` |
| pix2struct-docvqa-base_onnx_with_past (Quantized) | (88+183+169) MB = `440` MB   | `Encoder + Decoder + Decoder with Past` |

---

Testing Sample
- Image: data/documents/xfbc0228_2.png
- Question: When is the Memorandum dated on ?
- Ground truth: ['March 2, 1951']
- Predicted answer: March 2, 1951 (predicted same by all models)

---

Best Time taken by each model on Testing sample [Ran on Google Colab CPU for 5 times]:

| Model                                             | Initialization Time (sec) | Inference Time (sec)|
| ------------------------------------------------- | ------------------------- | ------------------- |
| pix2struct-docvqa-base                            | `2.84`                    | `29.70`             |
| pix2struct-docvqa-base_onnx                       | `2.14`                    | `28.37`             |
| pix2struct-docvqa-base_onnx (Quantized)           | `NaN`                     | `NaN`               |
| pix2struct-docvqa-base_onnx_with_past             | `3.79`                    | `17.5`              |
| pix2struct-docvqa-base_onnx_with_past (Quantized) | `NaN`                     | `NaN`               |

---