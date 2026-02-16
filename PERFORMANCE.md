# Performance Metrics

## Model Accuracy
CNN achieves high accuracy on validation data with strong confidence separation between AI and human speech.

## Inference Time
Average inference per audio: ~200–400 ms (cloud deployment).

## Confidence Scores
Typical confidence ranges:
- AI Generated: 0.90 – 0.99
- Human: 0.85 – 0.98

## Scalability
FastAPI supports concurrent requests and horizontal scaling.

## Optimization
- Lightweight CNN
- Fixed spectrogram size
- Batch inference support
