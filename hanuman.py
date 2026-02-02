# dynamic_recitation_corrected.py
import sounddevice as sd
import numpy as np
import whisper
from rapidfuzz import fuzz
import queue
import threading

# -------------------------------
# Configuration
# -------------------------------
MODEL_NAME = "medium"  # whisper model
SAMPLERATE = 16000
BLOCKSIZE = 16000  # 1 sec chunk
MATCH_THRESHOLD = 70  # fuzzy matching threshold (0-100)

# Split reference into lines
REFERENCE_TEXT = """
श्री हनुमान चालीसा किक पं दोहा श्रीगुरु चरन सरोज रज निज मनु मुकुरु सुधारि।
बरनऊँ रघुबर बिमल जसु जो दायकु फल चारि बुद्धिहीन तनु जानिके सुमिरों पवनकुमार।
बल बुद्धि बिद्या देहु मोहिं हरहु कलेस बिकार जय हनुमान ज्ञान गुन सागर।
जय कपीस तिहुँ लोक उजागर राम दूत अतुलित बल धामा।
अंजनिपुत्र पवनसुत नामा महाबीर बिक्रम बजरंगी।
""".strip().split("\n")

# -------------------------------
# Globals
# -------------------------------
audio_queue = queue.Queue()
stop_flag = threading.Event()
current_line_index = 0
model = whisper.load_model(MODEL_NAME)

# -------------------------------
# Audio callback
# -------------------------------
def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# -------------------------------
# Match line
# -------------------------------
def match_line(recognized_text, line_text):
    recognized_text = recognized_text.replace("\n", " ").strip()
    line_text = line_text.replace("\n", " ").strip()
    score = fuzz.partial_ratio(recognized_text, line_text)
    return score

# -------------------------------
# Audio processing thread
# -------------------------------
def process_audio():
    global current_line_index
    buffer = np.zeros((0, 1), dtype=np.float32)
    while not stop_flag.is_set():
        try:
            chunk = audio_queue.get(timeout=1)
        except queue.Empty:
            continue
        buffer = np.concatenate((buffer, chunk), axis=0)

        # Process every 3 seconds
        if len(buffer) >= SAMPLERATE * 3:
            audio_input = buffer.flatten()
            buffer = np.zeros((0, 1), dtype=np.float32)
            
            # Whisper expects float32 [-1,1]
            result = model.transcribe(audio_input, fp16=False, language="hi")
            recognized_text = result["text"].strip()
            
            if current_line_index >= len(REFERENCE_TEXT):
                print("🎉 Recitation complete!")
                stop_flag.set()
                break

            line_text = REFERENCE_TEXT[current_line_index]
            score = match_line(recognized_text, line_text)

            if score >= MATCH_THRESHOLD:
                print(f"✅ MATCH [{score}%] Line {current_line_index+1}: {line_text}")
                current_line_index += 1
            else:
                print(f"❌ MISMATCH [{score}%] Line {current_line_index+1}: {recognized_text}")

# -------------------------------
# Main
# -------------------------------
def main():
    print("🎤 Start recitation. Press Ctrl+C to stop...")
    threading.Thread(target=process_audio, daemon=True).start()
    try:
        with sd.InputStream(channels=1, samplerate=SAMPLERATE,
                            blocksize=BLOCKSIZE, callback=callback):
            while not stop_flag.is_set():
                sd.sleep(500)
    except KeyboardInterrupt:
        stop_flag.set()
        print("\n🛑 Stopped.")

if __name__ == "__main__":
    main()
