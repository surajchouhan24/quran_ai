# import os
# import logging
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
# import whisper
# import pytesseract
# from PIL import Image
# import difflib

# # Setup logging to see errors in terminal
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Path setup
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# FRONTEND_PATH = os.path.join(ROOT_DIR, "frontend")
# TEMP_PATH = os.path.join(ROOT_DIR, "temp")

# os.makedirs(TEMP_PATH, exist_ok=True)

# app = FastAPI(title="Quran AI Assistant")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load Whisper - using 'base' for speed, consider 'small' for better Arabic
# logger.info("Loading AI Model...")
# model = whisper.load_model("base")

# current_reference_text = ""

# @app.get("/")
# async def read_index():
#     index_path = os.path.join(FRONTEND_PATH, 'index.html')
#     if not os.path.exists(index_path):
#         return {"error": f"index.html not found at {index_path}"}
#     return FileResponse(index_path)

# # Important: Mount this AFTER the root route
# app.mount("/frontend", StaticFiles(directory=FRONTEND_PATH), name="frontend")

# @app.post("/upload-page")
# async def upload_page(file: UploadFile = File(...)):
#     global current_reference_text
#     file_path = os.path.join(TEMP_PATH, file.filename)
    
#     try:
#         content = await file.read()
#         with open(file_path, "wb") as f:
#             f.write(content)
        
#         image = Image.open(file_path)
#         # Using both ara (Arabic) and script/Arabic if available
#         extracted_text = pytesseract.image_to_string(image, lang='ara')
        
#         # Debugging: Print to terminal to see what OCR found
#         logger.info(f"OCR Extracted: {extracted_text}")
        
#         current_reference_text = " ".join(extracted_text.split())
        
#         if not current_reference_text.strip():
#             raise HTTPException(status_code=400, detail="OCR failed to find Arabic text.")
            
#         return {"reference": current_reference_text}
#     except Exception as e:
#         logger.error(f"Upload Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/analyze-recitation")
# async def analyze_recitation(file: UploadFile = File(...)):
#     global current_reference_text
#     if not current_reference_text:
#         raise HTTPException(status_code=400, detail="Upload a Quran page image first.")

#     audio_path = os.path.join(TEMP_PATH, "recitation.wav")
    
#     try:
#         content = await file.read()
#         with open(audio_path, "wb") as f:
#             f.write(content)

#         # Transcribe
#         # Task 'transcribe' ensures it stays in Arabic, doesn't translate to English
#         result = model.transcribe(audio_path, language="ar", task="transcribe")
#         spoken_text = result['text'].strip()
#         logger.info(f"User Recited: {spoken_text}")

#         ref_words = current_reference_text.split()
#         spoken_words = spoken_text.split()
        
#         feedback = []
#         for i, ref_word in enumerate(ref_words):
#             if i < len(spoken_words):
#                 # Similarity 0.7 is safe for ASR variations
#                 sim = difflib.SequenceMatcher(None, spoken_words[i], ref_word).ratio()
#                 status = "correct" if sim > 0.6 else "incorrect"
#                 feedback.append({"word": ref_word, "status": status, "said": spoken_words[i]})
#             else:
#                 feedback.append({"word": ref_word, "status": "missing"})

#         return {
#             "transcription": spoken_text,
#             "feedback": feedback,
#             "is_perfect": all(f['status'] == 'correct' for f in feedback)
#         }
#     except Exception as e:
#         logger.error(f"Analysis Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
# from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import HTMLResponse
# from pdf2image import convert_from_bytes
# import pytesseract
# import whisper
# import torch
# import numpy as np
# import re
# from rapidfuzz import fuzz
# from pathlib import Path
# import logging
# import asyncio
# # --------------------
# # Logging setup
# # --------------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("quran_ai")

# # --------------------
# # APP
# # --------------------
# app = FastAPI(title="Quran AI Assistant")

# BASE_DIR = Path(__file__).resolve().parent.parent
# FRONTEND_DIR = BASE_DIR / "frontend"

# app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# @app.get("/", response_class=HTMLResponse)
# def index():
#     logger.info("Serving index.html")
#     return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

# # --------------------
# # MODEL
# # --------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
# logger.info(f"Using device: {device}")
# whisper_model = whisper.load_model("small").to(device)

# REFERENCE_LINES = []
# current_line = 0
# MATCH_THRESHOLD = 72
# LANG = "hi"

# # --------------------
# # PDF → OCR (first page, remove header)
# # --------------------
# @app.post("/extract")
# async def extract(file: UploadFile = File(...)):
#     global REFERENCE_LINES, current_line
#     current_line = 0

#     logger.info(f"Received file: {file.filename}")
#     pdf_bytes = await file.read()
#     logger.info(f"PDF size: {len(pdf_bytes)} bytes")

#     try:
#         images = convert_from_bytes(pdf_bytes, dpi=300, first_page=1, last_page=1)
#         page = images[0]
#         w, h = page.size
#         page = page.crop((0, int(h * 0.18), w, h))
#         page = page.convert("L")

#         raw_text = pytesseract.image_to_string(page, lang="hin", config="--psm 6")
#         logger.info(f"OCR text length: {len(raw_text)}")

#         lines = []
#         for line in raw_text.splitlines():
#             line = re.sub(r"[^\u0900-\u097F\s]", "", line)
#             line = re.sub(r"\s+", " ", line).strip()
#             if len(line.split()) >= 2:  # relax for short Hindi lines
#                 lines.append(line)

#         REFERENCE_LINES = lines
#         logger.info(f"Lines stored: {len(lines)}")
#         logger.info(f"First 5 lines: {lines[:5]}")

#         return {"status": "ok", "total_lines": len(lines), "preview": lines[:3]}

#     except Exception as e:
#         logger.exception("PDF extraction failed")
#         return {"status": "error", "error": str(e)}
# # --- Update your MODEL section ---
# # Switching to 'base' for faster real-time inference
# whisper_model = whisper.load_model("base").to(device)
# @app.websocket("/listen")
# async def listen(ws: WebSocket):
#     # Accept the connection first
#     await ws.accept()
#     logger.info("🎤 WebSocket connected - High Speed Mode")
    
#     audio_buffer = np.array([], dtype=np.float32)
#     PROCESS_LIMIT = 12000  # ~0.75 seconds of audio

#     try:
#         while True:
#             # Receive data with a safety catch
#             try:
#                 data = await ws.receive_bytes()
#             except WebSocketDisconnect:
#                 logger.info("Client closed the connection.")
#                 break # Exit the while loop
                
#             chunk = np.frombuffer(data, dtype=np.float32)
#             audio_buffer = np.append(audio_buffer, chunk)

#             if len(audio_buffer) >= PROCESS_LIMIT:
#                 # Use beam_size=1 for speed; initial_prompt is removed as requested
#                 result = await asyncio.to_thread(
#                     whisper_model.transcribe, 
#                     audio_buffer, 
#                     language=LANG,
#                     beam_size=1,
#                     best_of=1,
#                     fp16=torch.cuda.is_available()
#                 )
                
#                 spoken = result["text"].strip()
#                 # COMPLETELY RESET BUFFER to stop hallucinations
#                 audio_buffer = np.array([], dtype=np.float32) 

#                 if spoken:
#                     logger.info(f"Detected: {spoken}")
#                     # Only send if the websocket is still open
#                     await ws.send_json({"type": "LIVE_DATA", "text": spoken})

#     except Exception as e:
#         logger.error(f"WebSocket Loop Error: {e}")
#     finally:
#         # This block ensures the session is logged as closed without crashing
#         logger.info("🎤 Session Closed")
#         # Don't call ws.accept() here!

# from fastapi import FastAPI, UploadFile, File
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import HTMLResponse
# from pdf2image import convert_from_bytes
# from PIL import ImageEnhance
# import pytesseract
# import re
# from pathlib import Path
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("quran_ai")

# app = FastAPI(title="Quran AI Assistant")

# # Setup paths
# BASE_DIR = Path(__file__).resolve().parent.parent # Add another .parent to go up
# FRONTEND_DIR = BASE_DIR / "frontend"

# # Then keep the rest as is
# app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# @app.get("/", response_class=HTMLResponse)
# def index():
#     return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")


# @app.post("/extract")
# async def extract(file: UploadFile = File(...)):
#     try:
#         pdf_bytes = await file.read()
#         # Extract first page
#         images = convert_from_bytes(pdf_bytes, dpi=300, first_page=1, last_page=1)
#         page = images[0]
        
#         # Crop header (approx 18%)
#         w, h = page.size
#         page = page.crop((0, int(h * 0.18), w, h)).convert("L")
#         enhancer = ImageEnhance.Contrast(page)
#         page = enhancer.enhance(2.0)
#         custom_config = r'--oem 1 --psm 6'
#         # OCR for Hindi/Sanskrit/Arabic-style scripts
#         raw_text = pytesseract.image_to_string(page, lang="hin", config="--psm 6")
        
#         # Clean text: keep only script characters and spaces
#         clean_text = re.sub(r"[^\u0900-\u097F\s]", "", raw_text)
#         words = clean_text.split()
#         print(f"OCR Extracted {len(words)} words")
#         print(f"OCR Extracted  Text: {words} ")

#         return {"status": "ok", "words": words}
#     except Exception as e:
#         logger.error(f"Error: {e}")
#         return {"status": "error", "error": str(e)}

import os
import io
import logging
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pdf2image import convert_from_bytes
from fastapi import HTTPException

from pathlib import Path
from dotenv import load_dotenv  # <-- Add this
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quran_ai")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY not found in environment variables!")
else:
    genai.configure(api_key=api_key)
# Use your key - recommend moving to environment variable later
# Use gemini-1.5-flash or gemini-2.0-flash-exp for high speed and accuracy
model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI(title="Hanuman Chalisa AI")

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

# @app.post("/extract")
# async def extract(file: UploadFile = File(...)):
#     try:
#         pdf_bytes = await file.read()
#         images = convert_from_bytes(pdf_bytes, dpi=300, first_page=1, last_page=1)
#         page = images[0]

#         # Convert to bytes for Gemini
#         img_byte_arr = io.BytesIO()
#         page.save(img_byte_arr, format='JPEG')
#         img_bytes = img_byte_arr.getvalue()

#         # prompt = """
#         # Extract the Arabic text from this image.
#         # - Correct any OCR spelling errors based on context.
#         # - Maintain the original verse structure (Ayaat/Couplets).
#         # - Return ONLY the Arabic text without any explanations.
#         # """
#         prompt = """
#             Extract the main Arabic text from this image while following these rules:
#             1. EXCLUDE all Headers, Footers, Page Numbers, and Marginalia.
#             2. Maintain the original visual structure of the verses (Ayaat/Couplets).
#             3. Correct OCR spelling errors based on Arabic context.
#             4. Return ONLY the main body text. No explanations, no markdown, and no layout descriptions.
#         """
#         response = model.generate_content([
#             prompt,
#             {"mime_type": "image/jpeg", "data": img_bytes}
#         ])
        
#         print(f"Gemini Raw Response: {response.text}")
        
#         # Clean text and split into words for the frontend's tracking logic
#         text_content = response.text.strip()
#         # Remove special characters but keep Hindi script
#         words = text_content.split()
        
#         print(f"Gemini Extracted {len(words)} words")
#         print(f"Gemini Extracted Text: {text_content}")

#         return {"status": "ok", "words": words, "raw_text": text_content}

#     except Exception as e:
#         logger.error(f"Error: {e}")
#         return {"status": "error", "error": str(e)}
@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise ValueError("Uploaded file is empty")

        images = convert_from_bytes(
            pdf_bytes,
            dpi=300,
            first_page=1,
            last_page=1
        )

        if not images:
            raise ValueError("No images extracted from PDF")

        page = images[0]

        img_byte_arr = io.BytesIO()
        page.save(img_byte_arr, format="JPEG")
        img_bytes = img_byte_arr.getvalue()

        prompt = """
        Extract the main Arabic text from this image while following these rules:
        1. EXCLUDE all Headers, Footers, Page Numbers, and Marginalia.
        2. Maintain the original visual structure of the verses (Ayaat/Couplets).
        3. Correct OCR spelling errors based on Arabic context.
        4. Return ONLY the main body text.
        """

        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": img_bytes}
        ])

        if not response or not response.text:
            raise ValueError("Gemini returned empty response")

        text_content = response.text.strip()
        words = text_content.split()

        logger.info(f"Gemini Extracted {len(words)} words")

        return {
            "status": "ok",
            "words": words,
            "raw_text": text_content
        }

    except Exception as e:
        logger.exception("Extraction failed")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )