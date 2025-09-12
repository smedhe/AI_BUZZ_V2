# Introduction to Efficient Transformers
## Overview
The Efficient Transformers library is a collection of optimized transformer models for various tasks. It provides a set of pre-trained models, including Llama4, Gemma3, and HP-CAI Grok-1, among others. The library allows for efficient inference and compilation of these models, with features such as sentence embedding, flexible pooling configuration, and support for multiple sequence lengths.

## Key Components / Concepts
The library consists of several key components and concepts, including:

*   **QEFFAutoModel**: A parent class for models provided by QEfficient, which includes AutoModel, AutoModelForCausalLM, and AutoModelForAudioClassification.
*   **QEFFBaseModel**: A base class for all model classes, providing utility methods for child classes.
*   **QEFFTransformersBase**: A parent class for models provided by QEfficient, which includes AutoModel, AutoModelForCausalLM, and AutoModelForAudioClassification.
*   **transform**: A function that optimizes any kind of model for Cloud AI 100.
*   **transform_lm**: A function that replaces some Transformers torch.nn.Module layers for equivalent optimized modules for Cloud AI 100.

## How it Works
The library works by providing a set of pre-trained models that can be used for various tasks. The models can be optimized for Cloud AI 100 using the `transform` function, which replaces some of the torch.nn.Module layers with optimized implementations. The library also provides a set of utility methods for child classes, including `QEFFBaseModel` and `QEFFTransformersBase`.

## Example(s)
To use the library, you can import the `QEFFAutoModel` class and use the `from_pretrained` method to load a pre-trained model. For example:
```python
from QEfficient import QEFFAutoModel
from transformers import AutoTokenizer

model = QEFFAutoModel.from_pretrained("model_name")
```
## Diagram(s)
```mermaid
flowchart TB
    subgraph QEFFAutoModel
        direction LR
        QEFFAutoModel[QEFFAutoModel] --> QEFFBaseModel[QEFFBaseModel]
        QEFFBaseModel --> QEFFTransformersBase[QEFFTransformersBase]
        QEFFTransformersBase --> transform[transform]
        transform --> transform_lm[transform_lm]
        transform_lm --> optimized_model[optimized model]
    end
```
Caption: Overview of the Efficient Transformers library architecture.

## References
*   `[QEfficient/README.md](QEfficient/README.md)`
*   `[QEfficient/transformers/transform.py](QEfficient/transformers/transform.py)`
*   `[QEfficient/transformers/models/modeling_auto.py](QEfficient/transformers/models/modeling_auto.py)`