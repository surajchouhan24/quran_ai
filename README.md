# 📖 Quran AI Memorization Assistant – Proof of Concept (PoC)

## Overview
This project is a **Proof of Concept (PoC)** for an AI-powered Quran memorization and recitation assistant.

The system allows users to:
- Upload a Quran PDF
- Extract accurate Arabic text using a **multimodal LLM (Gemini Vision)**
- Recite verses live
- Receive **real-time, word-level feedback** (correct / incorrect)
- View end-of-session accuracy statistics

The PoC demonstrates how **AI vision + browser-based speech recognition** can be combined to assist Quran memorization.

---

## Key Objectives
- Accurate Arabic text extraction from PDFs
- Real-time recitation tracking
- Blind memorization mode (hidden text while reciting)
- Word-by-word correctness validation
- Session-level scoring and feedback

---

## Key Features
- 📄 **PDF Upload** (first-page extraction for PoC)
- 🤖 **AI-based Arabic OCR** using Gemini Vision
- 🎙 **Live Arabic Speech Recognition** (Web Speech API)
- 🧠 **Blind Recitation Mode**
- ✅ **Word-by-word correctness tracking**
- 📊 **Accuracy & error statistics**
- 🌐 **FastAPI backend + HTML/CSS/JavaScript frontend**

---

## Tech Stack

### Backend
- Python **3.10**
- FastAPI
- Google Gemini Vision (`gemini-2.5-flash`)
- pdf2image
- python-dotenv

### Frontend
- HTML5
- CSS3 (responsive UI)
- Vanilla JavaScript
- Web Speech API (`SpeechRecognition`)
- RTL (Right-to-Left) Arabic text rendering

---

## Project Structure
```bash
quran_ai_app/
│
├── backend/
│ └── main.py            # FastAPI app & AI extraction logic
│
├── frontend/
│ └── index.html         # UI + client-side logic
│
├── venv/                # Python virtual environment
├── requirements.txt
├── .env                 # Store credentials
└── README.md
```

## 🧩 Installation & Setup

### 1️⃣ Clone the repository
 ``` bash
git clone https://github.com/GemsEssence/AI_Portfolio.git
cd quran_ai_app
```
### Create and activate a virtual environment
``` bash
python3.10 -m venv venv
source venv/bin/activate        # (On Windows: venv\Scripts\activate)
```

### Install dependencies
``` bash
pip install -r requirements.txt
```

### Create a .env file and add your API key
``` bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### Running the App

Start the FastAPI development server:
``` bash
uvicorn backend.main:app --reload
```

### Now open your browser and visit:
👉 http://127.0.0.1:8000

