# 🚀 AI Resume Skill Gap Analyzer

An AI-powered Resume Intelligence Platform that analyzes resumes against job descriptions, calculates skill match percentage, identifies skill gaps, and generates personalized learning suggestions and interview questions.

---

## 📌 Features
Features

📄 Upload Resume (PDF)

🧠 Automatic Skill Extraction

📊 Job Description Matching

📈 Job Match Score Calculation

🏢 ATS Compatibility Scoring (Weighted Model)

❌ Skill Gap Detection

📚 Learning Suggestions for Missing Skills

🎯 Interview Question Generator

⚠ ATS Issue Detection

🚀 ATS Improvement Suggestions

🌐 Professional React Web Interface

🔗 FastAPI + React Integration

---

## 🛠 Tech Stack

- Python
- FastAPI
- PyMuPDF (PDF Text Extraction)
- HTML + CSS (Inline UI)
- Uvicorn (ASGI Server)

---

## 📂 Project Structure

```bash

python -m venv venv
venv\Scripts\activate

pip install fastapi uvicorn pymupdf spacy python-multipart
python -m spacy download en_core_web_sm

uvicorn main:app --reload
http://127.0.0.1:8000/docs


