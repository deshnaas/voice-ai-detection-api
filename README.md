# AI Generated Voice Detection System

## Overview
This project detects whether an input voice sample is AI-generated or human-generated using deep learning on audio spectrograms. The system is exposed as a secure REST API.

The model converts audio into Mel Spectrogram images and uses a Convolutional Neural Network (CNN) to classify speech patterns.

## Features
- Accepts Base64 encoded MP3 audio
- Supports Tamil, English, Hindi, Malayalam, Telugu
- Returns classification with confidence score
- Secure API key authentication
- Deep learning based spectral analysis
- Production-ready FastAPI deployment

## Architecture
Audio Input (MP3 Base64)
→ Audio Decoding
→ Mel Spectrogram Extraction
→ CNN Classification
→ Confidence Scoring
→ JSON Response

## Tech Stack
- Python
- TensorFlow / Keras
- Librosa
- FastAPI
- NumPy
- Kaggle
- Railway Cloud

## Setup Instructions

### Install dependencies
