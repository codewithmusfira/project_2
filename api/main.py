from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from PIL import Image
import pytesseract
import joblib
import io
import os


# ---------------- INITIALIZE FASTAPI ----------------

app = FastAPI(
    title='Document Intelligence API',
    description='OCR, Classification, and Extraction',
    version='1.0.0'
)


# ---------------- LOAD MODELS ----------------

vectorizer_path = 'models/vectorizer.pkl'
classifier_path = 'models/classifier.pkl'

if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Missing: {vectorizer_path}")

if not os.path.exists(classifier_path):
    raise FileNotFoundError(f"Missing: {classifier_path}")

vectorizer = joblib.load(vectorizer_path)
classifier = joblib.load(classifier_path)


# ---------------- ROOT ENDPOINT ----------------

@app.get('/')
def root():
    return {
        'message': 'Document Intelligence API',
        'version': '1.0.0',
        'endpoints': [
            '/classify',
            '/extract',
            '/process'
        ]
    }
