# AI-VS-HUMAN

This project is intended to detect if a given text has been generated by a machine or a human

# AI-Generated Text Detection

## Overview

This project aims to distinguish between human-written and AI-generated text using a deep learning approach. Given the increasing sophistication of language models, it is crucial to develop methods capable of identifying synthetic text to ensure transparency and authenticity in digital content.

## Approach

We utilize **DistilBERT**, a lightweight version of BERT, to generate contextual embeddings from input text. These embeddings serve as rich numerical representations of the text, capturing deep semantic relationships. Instead of relying on traditional statistical classifiers, which assume independent and identically distributed (i.i.d) vectors, we leverage a neural network to learn complex patterns that differentiate human and AI-generated text.

### Data Representation

Each input text is converted into a sequence of token embeddings. To obtain a fixed-size vector representation, we compute the mean of all token embeddings:

![](Images/absolute.png)

where alpha_i represents the contextual embedding of token i.

The dataset is then structured as ordered pairs:

![](Images/order_pair.png)

where L represents the binary label indicating whether the text is human-written (0) or AI-generated (1).

### Model Architecture

A **fully connected neural network** is trained on these embeddings to perform classification. The architecture consists of:

- **Input layer**: 768-dimensional embedding vector
- **Hidden layer**: 256 neurons with ReLU activation
- **Output layer**: 2 neurons with softmax activation for binary classification

### Training and Evaluation

- **Dataset Split**: 80% training, 20% testing
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score

## Results

The model successfully differentiates between human-written and AI-generated text with high accuracy, demonstrating the effectiveness of using transformer-based embeddings combined with a deep learning classifier.

## Future Work

- Experimenting with larger transformer models like BERT or RoBERTa for improved feature extraction.
- Exploring alternative classification architectures such as LSTMs or CNNs for enhanced pattern recognition.
- Fine-tuning on domain-specific datasets for tailored detection capabilities.

## Usage

It's available on :https://ai-vs-human-6qv7.onrender.com
