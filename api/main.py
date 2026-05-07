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
@app.post('/classify')
async def classify_document(file: UploadFile = File(...)):
    """
    Classify document type

    Returns:
        document type and confidence
    """

    try:
        # ---------------- READ IMAGE ----------------

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        )


        # ---------------- OCR ----------------

        text = pytesseract.image_to_string(image)


        # ---------------- VECTORIZE ----------------

        text_vec = vectorizer.transform([text])


        # ---------------- PREDICTION ----------------

        prediction = classifier.predict(text_vec)[0]

        probabilities = classifier.predict_proba(text_vec)[0]

        confidence = max(probabilities)


        # ---------------- RESPONSE ----------------

        return {
            'document_type': prediction,

            'confidence': float(confidence),

            'all_probabilities': {
                cls: float(prob)
                for cls, prob in zip(
                    classifier.classes_,
                    probabilities
                )
            }
        }


    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                'error': str(e)
            }
        )
        # ---------------- IMPORT EXTRACTION FUNCTIONS ----------------

import sys
import os

sys.path.append(os.path.abspath(".."))

from week7_extraction import (
    extract_dates,
    extract_amounts,
    extract_entities
)


# ---------------- EXTRACT ENDPOINT ----------------

@app.post('/extract')
async def extract_information(file: UploadFile = File(...)):
    """
    Extract information from document

    Returns:
        extracted dates, amounts, entities
    """

    try:
        # ---------------- READ IMAGE ----------------

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        )


        # ---------------- OCR ----------------

        text = pytesseract.image_to_string(image)


        # ---------------- EXTRACTION ----------------

        dates = extract_dates(text)

        amounts = extract_amounts(text)

        entities = extract_entities(text)


        # ---------------- RESPONSE ----------------

        return {
            'dates': dates,

            'amounts': amounts,

            'entities': entities,

            'raw_text': text[:500]
        }


    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                'error': str(e)
            }
        )
        @app.post('/process')
async def process_document(file: UploadFile = File(...)):
    """
    Complete pipeline:
    OCR + Classification + Information Extraction

    Returns:
        document type and extracted data
    """

    try:
        # ---------------- READ IMAGE ----------------

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        )


        # ---------------- OCR ----------------

        text = pytesseract.image_to_string(image)


        # ---------------- CLASSIFICATION ----------------

        text_vec = vectorizer.transform([text])

        doc_type = classifier.predict(text_vec)[0]

        probabilities = classifier.predict_proba(text_vec)[0]

        confidence = max(probabilities)


        # ---------------- EXTRACTION ----------------

        extracted_data = {
            'dates': extract_dates(text),

            'amounts': extract_amounts(text),

            'entities': extract_entities(text)
        }


        # ---------------- RESPONSE ----------------

        return {
            'document_type': doc_type,

            'confidence': float(confidence),

            'extracted_data': extracted_data,

            'status': 'success'
        }


    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                'error': str(e),
                'status': 'failed'
            }
        )
        
