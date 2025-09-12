# Introduction to Efficient Transformers
## Overview
The Efficient Transformers Library is a collection of efficient transformer models and tools for natural language processing tasks. It provides a range of features, including sentence embedding, flexible pooling configuration, and support for multiple sequence lengths.

## Key Components / Concepts
The library includes several key components and concepts, including:

*   QEFFAutoModel: A class for manipulating any transformer model from the HuggingFace hub.
*   QEFFAutoModelForCausalLM: A class for manipulating any causal language model from the HuggingFace hub.
*   QEFFAutoModelForSpeechSeq2Seq: A class for transformers models with a sequence-to-sequence speech-to-text modeling head.
*   QEFFAutoModelForImageTextToText: A class for working with multimodal language models from the HuggingFace hub.
*   transform: A function for optimizing any kind of model (i.e. LLM, SD, AWQ etc.) for Cloud AI 100.
*   transform_lm: A function for replacing some Transformers torch.nn.Module layers for equivalent optimized modules for Cloud AI 100.

## How it Works
The library works by providing a range of tools and features for efficient inference and compilation of transformer models. It supports various model execution modes, including QNN compilation, SwiftKV, and gradient checkpointing.

## Example(s)
Here is an example of how to use the QEFFAutoModel class to load a pre-trained model and compile it for Cloud AI 100:
```python
from QEfficient import QEFFAutoModel
from transformers import AutoTokenizer

# Initialize the model using from_pretrained similar to transformers.AutoModel.
model = QEFFAutoModel.from_pretrained("model_name")

# Now you can directly compile the model for Cloud AI 100
model.compile(num_cores=16)  # Considering you have a Cloud AI 100 SKU

# Prepare input
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("My name is", return_tensors="pt")

# You can now execute the model
model.generate(inputs)
```

## Diagram(s)
```mermaid
graph LR
    A[QEFFAutoModel] -->|load pre-trained model|> B[QEFFAutoModelForCausalLM]
    B -->|compile for Cloud AI 100|> C[Compiled Model]
    C -->|execute model|> D[Output]
```
Caption: Overview of the Efficient Transformers Library

## References
*   [QEfficient/transformers/models/modeling_auto.py](https://github.com/quic/efficient-transformers/blob/main/QEfficient/transformers/models/modeling_auto.py)
*   [QEfficient/transformers/transform.py](https://github.com/quic/efficient-transformers/blob/main/QEfficient/transformers/transform.py)
*   [README.md](https://github.com/quic/efficient-transformers/blob/main/README.md)