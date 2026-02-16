# Model Overview

## Input Representation
Audio is converted into Mel Spectrogram images (128x128).

## Architecture
- Conv2D Layers
- MaxPooling
- Dense Layers
- Softmax Output

## Classification Logic
The model detects:

- Spectral smoothness
- Pitch consistency
- Temporal regularity

These artifacts are typical of synthetic speech.

## Confidence Score
Confidence is derived from softmax probability.

## Training
Model trained on labeled AI and human voice samples using supervised learning.

## Evaluation
Accuracy and confusion matrix used for validation.
