# Model Integration
## Overview
The Efficient Transformers Library provides a simple and efficient way to integrate models into your applications. The library offers a range of features and tools to help you get started with model integration, including model loading, compilation, and execution.

## Key Components / Concepts
The key components and concepts involved in model integration with the Efficient Transformers Library are:

*   **Model Loading**: The library provides a simple and efficient way to load pre-trained models from the Hugging Face model hub or from local directories.
*   **Model Compilation**: The library allows you to compile models for Cloud AI 100, which enables efficient execution on the Cloud AI 100 platform.
*   **Model Execution**: The library provides a range of tools and APIs to execute models, including support for continuous batching and speculative decoding.

## How it Works
The model integration process with the Efficient Transformers Library involves the following steps:

1.  **Model Loading**: Load a pre-trained model from the Hugging Face model hub or from a local directory using the `from_pretrained` method.
2.  **Model Compilation**: Compile the loaded model for Cloud AI 100 using the `compile` method.
3.  **Model Execution**: Execute the compiled model using the `generate` method.

## Example(s)
Here is an example of how to integrate a model into your application using the Efficient Transformers Library:

```python
from QEfficient import QEFFAutoModel
from transformers import AutoTokenizer

# Load a pre-trained model from the Hugging Face model hub
model = QEFFAutoModel.from_pretrained("model_name")

# Compile the model for Cloud AI 100
model.compile(num_cores=16)

# Prepare input
tokenizer = AutoTokenizer.from_pretrained("model_name")
inputs = tokenizer("My name is", return_tensors="pt")

# Execute the model
model.generate(inputs)
```

## Diagram(s)
```mermaid
graph LR
    A[Model Loading] --> B[Model Compilation]
    B --> C[Model Execution]
    C --> D[Output]
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#f9f,stroke:#333,stroke-width:2px
    style A stroke:#333,stroke-width:2px
    style B stroke:#333,stroke-width:2px
    style C stroke:#333,stroke-width:2px
    style D stroke:#333,stroke-width:2px
    note right of A: Load a pre-trained model from the Hugging Face model hub
    note right of B: Compile the loaded model for Cloud AI 100
    note right of C: Execute the compiled model
    note right of D: Get the output of the model
```

## References
*   [QEfficient/transformers/models/modeling_auto.py](https://github.com/your-repo/QEfficient/blob/main/transformers/models/modeling_auto.py)
*   [QEfficient/transformers/transform.py](https://github.com/your-repo/QEfficient/blob/main/transformers/transform.py)
*   [QEfficient/utils/_utils.py](https://github.com/your-repo/QEfficient/blob/main/utils/_utils.py)