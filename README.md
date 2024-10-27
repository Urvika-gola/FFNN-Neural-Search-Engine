# Dependency Parsing with Feedforward Neural Network and FancyArcStandardParser

## Overview
This project builds a dependency parser using a **Feedforward Neural Network** model and an **Arc-Standard Parsing** approach via the `FancyArcStandardParser` class. The parser utilizes syntactic features from sentences to predict dependency relations (heads and labels) for each token in a sentence. The model can handle standard dependency parsing transitions: `Shift`, `Left Arc`, and `Right Arc`.

Developed for **Advanced NLP and Parsing Techniques**.

## Features

- **Feedforward Neural Network (FFNN)**:
  - Three-layer FFNN with ReLU activations and dropout for regularization.
  - Trained on vectorized features of the stack and buffer states of the parser.
  - Outputs transition probabilities for parsing decisions.

- **FancyArcStandardParser**:
  - **Arc-Standard Parsing**: Utilizes the Arc-Standard transition system with `Shift`, `Left Arc`, and `Right Arc` operations.
  - **UPOS Feature Vectorization**: Converts the stack and buffer configurations into fixed-size feature vectors for the FFNN.
  - **Viterbi-like Parsing**: Integrates neural predictions with heuristic rules for unseen words and transitions.
