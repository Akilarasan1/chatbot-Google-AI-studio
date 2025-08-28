# chatbot-Google-AI-studio

A Streamlit-powered AI chatbot that extracts text from PDFs and images using OCR and enables users to interact with the extracted content via Google Gemini AI.

✨ Features

✔️ Upload PDF and image files for text extraction

✔️ OCR processing using Google Cloud Vision API & EasyOCR

✔️ Chunking long text for efficient querying

✔️ Semantic search using FAISS vector store

✔️ Conversational AI with Gemini 2.0 Flash

✔️ Streamlit-based UI for easy interaction

🛠 Tech Stack
Python

Streamlit

Google Cloud Vision API for OCR
EasyOCR for text extraction
FAISS for efficient search
Google Gemini AI for Q&A

LangChain for LLM interaction

🚀 Setup & Installation
GOOGLE_API_KEY=your-google-api-key

Run the application:
    streamlit run app.py
    
📝 How It Works
  Upload PDFs & images
  Extract text using OCR
  Store data in FAISS vector DB
  Ask questions from extracted text
  Get AI-generated answers
