import re
import pickle
import numpy as np

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# Paths
# ============================================================

model_path = "Artifacts/bigru.keras"
tokenizer_path = "Artifacts/tokenizer.pkl"

max_sequence_length = 50


# ============================================================
# Emotion labels
# ============================================================

emotion_labels = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]


# ============================================================
# Emotion emojis
# ============================================================

emotion_emojis = {
    "sadness": "😔",
    "joy": "🤩",
    "love": "😍",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😦"
}


# ============================================================
# Text preprocessing
# ============================================================

def preprocess_text(text: str) -> str:
    text = text.lower()

    # Remove apostrophes
    text = re.sub(r"'", "", text)

    # Keep only letters, numbers and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# Request schema
# ============================================================

class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={
            "example": "I feel so happy and excited"
        }
    )


# ============================================================
# Response schemas
# ============================================================

class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ============================================================
# Model storage
# ============================================================

dl_model = {}


# ============================================================
# Lifespan - load model and tokenizer
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading the model and tokenizer...")

    try:
        # Load BiGRU model
        dl_model["bigru"] = load_model(model_path)

        # Load tokenizer
        with open(tokenizer_path, "rb") as file:
            dl_model["tokenizer"] = pickle.load(file)

        print("Model and tokenizer loaded successfully!")

        yield

    except Exception as e:
        print(f"Error while loading model/tokenizer: {e}")

        # Still allow FastAPI to start
        yield

    finally:
        dl_model.clear()
        print("Model resources cleared.")


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Emotion Detection API",
    description="BiGRU based emotion detection API",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# Static files
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ============================================================
# Home endpoint
# ============================================================

@app.get("/", include_in_schema=False)
def server_ui():

    return FileResponse(
        "static/index.html"
    )


# ============================================================
# Health check endpoint
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health_check():

    return HealthResponse(
        status="Server is running",
        model_loaded=bool(dl_model)
    )


# ============================================================
# Prediction endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_emotion(text_input: TextInput):

    # Get model and tokenizer
    bigru_model = dl_model.get("bigru")
    tokenizer_model = dl_model.get("tokenizer")

    # Check whether model is loaded
    if bigru_model is None or tokenizer_model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please try again later."
        )

    # --------------------------------------------------------
    # Preprocess text
    # --------------------------------------------------------

    cleaned_text = preprocess_text(
        text_input.text
    )

    # --------------------------------------------------------
    # Tokenize text
    # --------------------------------------------------------

    tokenized_text = tokenizer_model.texts_to_sequences(
        [cleaned_text]
    )

    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probabilities = bigru_model.predict(
        padded_sequence,
        verbose=0
    )[0]

    # --------------------------------------------------------
    # Get top emotion
    # --------------------------------------------------------

    top_emotion_index = int(
        np.argmax(probabilities)
    )

    predicted_emotion = emotion_labels[
        top_emotion_index
    ]

    confidence = float(
        probabilities[top_emotion_index]
    )

    # --------------------------------------------------------
    # All probabilities
    # --------------------------------------------------------

    all_probabilities = {
        label: float(prob)
        for label, prob in zip(
            emotion_labels,
            probabilities
        )
    }

    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=predicted_emotion,
        confidence=confidence,
        all_probabilities=all_probabilities
    )