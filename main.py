"""
AI Generated Voice Detection API

This FastAPI application provides a REST endpoint for detecting
whether an input MP3 audio sample is AI-generated or human speech.

Pipeline:
1. Base64 audio decoding
2. Mel Spectrogram extraction
3. CNN-based classification
4. Confidence scoring
5. Structured JSON response

Security:
- API Key authentication
- Input validation
- MP3-only enforcement

Author: [Your Name]
"""




from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import librosa
import tempfile
import os
import gdown
from tensorflow.keras.models import load_model

# ================= CONFIG =================
API_KEY = "sk_guvi_demo_123456"

SUPPORTED_LANGUAGES = {
    "english": "English",
    "tamil": "Tamil",
    "hindi": "Hindi",
    "malayalam": "Malayalam",
    "telugu": "Telugu"
}

MODEL_PATH = "voice_ai_cnn_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=15aXWpMUfQkRVbt4z8KzRWB7SGvjJiJUd&confirm=t"

# ================= LOAD MODEL =================
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10_000_000:
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

# ================= APP =================
app = FastAPI(title="AI Generated Voice Detection API")

# ================= REQUEST SCHEMA =================
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# ================= UTILS =================
def extract_mel_spectrogram(audio_bytes, n_mels=128, max_len=128):
    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        audio, sr = librosa.load(tmp.name, sr=None)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        if mel_db.shape[1] < max_len:
            mel_db = np.pad(
                mel_db,
                ((0, 0), (0, max_len - mel_db.shape[1])),
                mode="constant"
            )
        else:
            mel_db = mel_db[:, :max_len]

        return mel_db.reshape(1, 128, 128, 1)

def explain(classification):
    if classification == "AI_GENERATED":
        return "Unnatural spectral consistency and synthetic speech patterns detected"
    else:
        return "Natural pitch variation and human speech characteristics detected"

# ================= API ENDPOINT =================
@app.post("/api/voice-detection")
def detect_voice(
    request: VoiceRequest,
    x_api_key: str = Header(None)
):
    # API key validation
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Language validation (CASE-INSENSITIVE)
    language_input = request.language.strip().lower()
    if language_input not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    standardized_language = SUPPORTED_LANGUAGES[language_input]

    # Audio format validation
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format supported")

    # Base64 decode
    try:
        audio_bytes = base64.b64decode(request.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # Prediction
    features = extract_mel_spectrogram(audio_bytes)
    probs = model.predict(features)[0]

    confidence = float(np.max(probs))
    prediction = int(np.argmax(probs))
    classification = "AI_GENERATED" if prediction == 1 else "HUMAN"

    return {
        "status": "success",
        "language": standardized_language,
        "classification": classification,
        "confidenceScore": round(confidence, 3),
        "explanation": explain(classification)
    }

## Documentation Index

- API_DOCS.md
- MODEL_OVERVIEW.md
- SECURITY.md
- PERFORMANCE.md
- INSTALLATION.md
- USAGE.md
- ARCHITECTURE.md
- PROJECT_STRUCTURE.md

