# QEfficient Overview
## Overview
QEfficient is a library designed for efficient inference on Qualcomm Cloud AI 100 using Hugging Face Transformer models. It provides a simple and easy-to-use interface for loading, compiling, and generating models, as well as a set of utility functions for tasks such as quantization and model management.

## Key Components / Concepts
QEfficient consists of several key components and concepts, including:

*   **QEFFAutoModel**: A class for loading and manipulating Hugging Face Transformer models.
*   **QEFFAutoModelForCausalLM**: A class for loading and manipulating causal language models from the Hugging Face hub.
*   **QEFFAutoModelForImageTextToText**: A class for loading and manipulating multimodal language models from the Hugging Face hub.
*   **CloudAI100ExecInfo**: A class for holding information about Cloud AI 100 execution, including batch size, generated texts, generated IDs, and performance metrics.

## How it Works
QEfficient works by providing a simple and easy-to-use interface for loading, compiling, and generating models. The library uses the Hugging Face Transformer models and provides a set of utility functions for tasks such as quantization and model management.

## Example(s)
Here is an example of how to use QEfficient to load and compile a model:
```python
from QEfficient import QEFFAutoModel

model = QEFFAutoModel.from_pretrained("bert-base-uncased")
model.compile()
```
## Diagram(s)
```mermaid
graph LR
    A[QEfficient Library] --> B[Load Model]
    B --> C[Compile Model]
    C --> D[Generate Model]
    D --> E[Cloud AI 100 Execution]
    E --> F[Performance Metrics]
```
Caption: QEfficient Library Flowchart

## References
*   [QEfficient/__init__.py](QEfficient/__init__.py)
*   [QEfficient/utils/__init__.py](QEfficient/utils/__init__.py)
*   [docs/index.md](docs/index.md)