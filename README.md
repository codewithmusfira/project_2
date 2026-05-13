Document Intelligence API

A complete AI-powered document processing system built with Python and FastAPI. This project performs OCR, document classification, and information extraction from uploaded document images such as invoices, receipts, and contracts.

Features
OCR text extraction using Tesseract
Document classification using Machine Learning
TF-IDF text vectorization
Information extraction (dates, amounts, entities)
REST API built with FastAPI
Interactive Swagger API documentation
End-to-end document processing pipeline
Technologies Used
Python
FastAPI
Scikit-learn
Tesseract OCR
TF-IDF Vectorizer
Logistic Regression
Joblib
Pillow (PIL)
Project Workflow

Image Upload → OCR → Text Extraction → TF-IDF Vectorization → Document Classification → Information Extraction → JSON Response

Supported Document Types
Invoices
Receipts
Contracts
Project Structure
project/
│
├── training_data/
│   ├── invoices/
│   ├── receipts/
│   └── contracts/
│
├── models/
│   ├── vectorizer.pkl
│   └── classifier.pkl
│
├── api/
│   └── main.py
│
├── week8_classifier_and_api.ipynb
└── README.md
Installation

Clone the repository:

git clone <your-repo-link>
cd <repo-name>

Install required libraries:

pip install fastapi uvicorn python-multipart scikit-learn pytesseract pillow joblib
Run the API
cd api
uvicorn main:app --reload

Server will run at:

http://127.0.0.1:8000
API Endpoints
1. /classify

Classifies uploaded documents into invoice, receipt, or contract.

2. /extract

Extracts important information such as:

Dates
Amounts
Entities
3. /process

Complete pipeline:

OCR
Classification
Information Extraction
Swagger Documentation

Open the interactive API documentation:

http://127.0.0.1:8000/docs
Machine Learning Model
TF-IDF Vectorizer
Logistic Regression Classifier
Accuracy target: 90%+
Output Example
{
  "document_type": "invoice",
  "confidence": 0.96,
  "extracted_data": {
    "dates": ["12/05/2026"],
    "amounts": ["$250"],
    "entities": ["Company ABC"]
  },
  "status": "success"
}
Future Improvements
Add more document types
Deploy on cloud platforms
Improve OCR accuracy
Add deep learning models
Support PDF files
Author

Musfira Nazahat

License

This project is for educational and learning purposes.
