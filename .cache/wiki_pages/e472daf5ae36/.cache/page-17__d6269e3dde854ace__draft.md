# QEfficientGPT2 Notebook
## Overview
The QEfficientGPT2 notebook demonstrates the usage of the QEfficientGPT2 model, which is a variant of the GPT2 model optimized for efficient inference on Cloud AI 100.

## Key Components / Concepts
The QEfficientGPT2 model is built on top of the Hugging Face Transformers library and uses the QEfficient framework for efficient inference. The key components of the QEfficientGPT2 model include:

*   QEfficient framework: A framework for efficient inference on Cloud AI 100.
*   Hugging Face Transformers library: A library for building and training transformer models.
*   GPT2 model: A variant of the GPT model optimized for efficient inference.

## How it Works
The QEfficientGPT2 model works by first loading the pre-trained GPT2 model from the Hugging Face model hub. It then applies the QEfficient framework to optimize the model for efficient inference on Cloud AI 100. The optimized model is then compiled and executed on the Cloud AI 100 platform.

## Example(s)
```python
from QEfficient import QEFFAutoModelForCausalLM

# Load the pre-trained GPT2 model
model = QEFFAutoModelForCausalLM.from_pretrained("gpt2")

# Compile the model for Cloud AI 100
model.compile(num_cores=16)

# Prepare input
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer("My name is", return_tensors="pt")

# Execute the model
model.generate(inputs)
```

## Diagram(s)
```mermaid
graph LR
    A[Load pre-trained GPT2 model] --> B[Apply QEfficient framework]
    B --> C[Compile model for Cloud AI 100]
    C --> D[Execute model on Cloud AI 100]
```
Caption: QEfficientGPT2 Model Workflow

## References
*   [QEfficientGPT2.ipynb](notebooks/QEfficientGPT2.ipynb)
*   [QEfficient documentation](https://qefficient.readthedocs.io/en/latest/)
*   [Hugging Face Transformers library documentation](https://huggingface.co/docs/transformers/index)