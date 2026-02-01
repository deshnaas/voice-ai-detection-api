from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import librosa
import tempfile
from tensorflow.keras.models import load_model

# ================= CONFIG =================
API_KEY = "sk_guvi_demo_123456"
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]


import os
import gdown

MODEL_PATH = "voice_ai_cnn_model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/file/d/15aXWpMUfQkRVbt4z8KzRWB7SGvjJiJUd/view?usp=sharing"
    gdown.download(url, MODEL_PATH, quiet=False)

# ================= LOAD MODEL =================
model = load_model("voice_ai_cnn_model.h5")

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
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format supported")

    try:
        audio_bytes = base64.b64decode(request.audioBase64)
    except:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    features = extract_mel_spectrogram(audio_bytes)
    probs = model.predict(features)[0]

    confidence = float(np.max(probs))
    prediction = int(np.argmax(probs))

    classification = "AI_GENERATED" if prediction == 1 else "HUMAN"

    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": round(confidence, 3),
        "explanation": explain(classification)
    }
