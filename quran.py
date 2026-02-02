# import difflib
# import pytesseract
# import whisper
# import torch
# import sounddevice as sd
# import tempfile
# import scipy.io.wavfile as wav
# from pdf2image import convert_from_path
# import re

# # -----------------------------
# # CONFIG
# # -----------------------------
# PDF_PATH = "1-sri-hanuman-chalisa-devanagiri.pdf"
# RECORD_SECONDS = 10
# SAMPLE_RATE = 16000

# # -----------------------------
# # LOAD WHISPER MODEL
# # -----------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("🔁 Loading Whisper model...", device)
# asr_model = whisper.load_model("small").to(device)

# # -----------------------------
# # OCR FUNCTION
# # -----------------------------
# def extract_first_page_text():
#     """Extract Hindi/Sanskrit text exactly as-is from PDF"""
#     images = convert_from_path(PDF_PATH, dpi=300)
#     text = pytesseract.image_to_string(images[0], lang="hin")
#     return text  # no cleaning

# # -----------------------------
# # RECORD AUDIO
# # -----------------------------
# def record_audio():
#     print(f"\n🎤 Recording for {RECORD_SECONDS} seconds...")
#     audio = sd.rec(
#         int(RECORD_SECONDS * SAMPLE_RATE),
#         samplerate=SAMPLE_RATE,
#         channels=1,
#         dtype="int16"
#     )
#     sd.wait()
#     print("✅ Recording complete")
#     return audio

# # -----------------------------
# # SPEECH → TEXT
# # -----------------------------
# def speech_to_text(audio):
#     """Convert audio to text using Whisper"""
#     with tempfile.NamedTemporaryFile(suffix=".wav") as f:
#         wav.write(f.name, SAMPLE_RATE, audio)
#         result = asr_model.transcribe(f.name, language="hi")
#         text = result["text"]
#         text = re.sub(r"\s+", " ", text).strip()
#         return text

# # -----------------------------
# # COMPARE FUNCTION
# # -----------------------------
# def compare(reference, spoken):
#     """Compare spoken text with reference and show mistakes"""
#     ref_words = reference.split()
#     spoken_words = spoken.split()
#     matcher = difflib.SequenceMatcher(None, ref_words, spoken_words)

#     for tag, i1, i2, j1, j2 in matcher.get_opcodes():
#         if tag == "equal":
#             for w in ref_words[i1:i2]:
#                 print(f"✅ RIGHT: {w}")
#         elif tag == "replace":
#             print(f"❌ WRONG: {ref_words[i1:i2]} → You said {spoken_words[j1:j2]}")
#         elif tag == "delete":
#             print(f"⚠️ Missing: {ref_words[i1:i2]}")
#         elif tag == "insert":
#             print(f"➕ Extra said: {spoken_words[j1:j2]}")

# # -----------------------------
# # MAIN LOOP
# # -----------------------------
# if __name__ == "__main__":
#     print("📖 Extracting text from PDF...")
#     pdf_text = extract_first_page_text()
#     print("\n📜 Reference Text (first 500 chars):\n")
#     print(pdf_text[:500], "...\n")

#     while True:
#         choice = input("Press 'y' to start reciting, 'q' to quit: ")
#         if choice == 'q':
#             print("👋 Exiting.")
#             break
#         elif choice == 'y':
#             audio = record_audio()
#             spoken_text = speech_to_text(audio)
#             print("\n🗣️ You said:\n")
#             print(spoken_text)
#             print("\n📊 Comparison:\n")
#             compare(pdf_text, spoken_text)
#         else:
#             print("⚠️ Invalid input. Press 'y' or 'q'.")
import sounddevice as sd
import numpy as np
import whisper
import pytesseract
from pdf2image import convert_from_path
from rapidfuzz import fuzz
import threading
import queue
import time
import re
import torch

# ===============================
# CONFIG
# ===============================
PDF_PATH = "1-sri-hanuman-chalisa-devanagiri.pdf"
LANG = "hi"
MODEL_NAME = "small"
SAMPLE_RATE = 16000
BLOCKSIZE = 8000           # 0.5 sec
PROCESS_EVERY = 1.2        # seconds
MATCH_THRESHOLD = 72

# ===============================
# LOAD MODELS
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("🔁 Loading Whisper model...", device)
model = whisper.load_model(MODEL_NAME).to(device)

# ===============================
# OCR → CLEAN LINES
# ===============================
def extract_lines_from_pdf():
    images = convert_from_path(PDF_PATH, dpi=300)
    raw = pytesseract.image_to_string(images[0], lang="hin")

    lines = []
    for line in raw.splitlines():
        line = re.sub(r"[^\u0900-\u097F\s।॥]", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if len(line.split()) >= 3:
            lines.append(line.replace("।", "").replace("॥", ""))

    return lines

REFERENCE_LINES = extract_lines_from_pdf()

print("\n📜 Loaded Lines:")
for i, l in enumerate(REFERENCE_LINES[:5], 1):
    print(f"{i}. {l}")

# ===============================
# AUDIO QUEUE
# ===============================
audio_queue = queue.Queue()
stop_flag = threading.Event()
current_line = 0

# ===============================
# AUDIO CALLBACK
# ===============================
def callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# ===============================
# MATCH LOGIC (SEQUENCE SAFE)
# ===============================
def sequence_match(spoken, expected_line):
    spoken = spoken.strip()
    if len(spoken.split()) < 2:
        return 0
    return fuzz.partial_ratio(spoken, expected_line)

# ===============================
# PROCESS THREAD
# ===============================
def process_audio():
    global current_line
    buffer = np.zeros((0, 1), dtype=np.float32)
    last_time = time.time()

    while not stop_flag.is_set():
        try:
            chunk = audio_queue.get(timeout=0.5)
            buffer = np.concatenate((buffer, chunk))
        except queue.Empty:
            continue

        if time.time() - last_time < PROCESS_EVERY:
            continue

        last_time = time.time()

        if len(buffer) < SAMPLE_RATE:
            continue

        audio = buffer.flatten()
        buffer = np.zeros((0, 1), dtype=np.float32)

        result = model.transcribe(
            audio,
            language=LANG,
            fp16=False,
            condition_on_previous_text=False
        )

        spoken = result["text"].strip()
        if not spoken:
            continue

        if current_line >= len(REFERENCE_LINES):
            print("\n🎉 PAGE COMPLETED 🎉")
            stop_flag.set()
            return

        expected = REFERENCE_LINES[current_line]
        score = sequence_match(spoken, expected)

        if score >= MATCH_THRESHOLD:
            print(f"✅ Line {current_line+1} OK [{score:.1f}%]")
            print(f"   {expected}\n")
            current_line += 1
        else:
            print(f"❌ ERROR Line {current_line+1} [{score:.1f}%]")
            print(f"   You said: {spoken}\n")

# ===============================
# MAIN
# ===============================
def main():
    print("\n🎤 REAL-TIME RECITATION STARTED")
    print("📖 Read line by line (sequence enforced)")
    print("🛑 Ctrl+C to stop\n")

    threading.Thread(target=process_audio, daemon=True).start()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=BLOCKSIZE,
            callback=callback
        ):
            while not stop_flag.is_set():
                time.sleep(0.1)
    except KeyboardInterrupt:
        stop_flag.set()
        print("\n🛑 Stopped")

# ===============================
if __name__ == "__main__":
    main()

