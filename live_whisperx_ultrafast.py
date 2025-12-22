import sounddevice as sd
import numpy as np
import webrtcvad
import torch
import threading
import queue
import time
import sys
import ollama
import asyncio
from faster_whisper import WhisperModel
from kittentts import KittenTTS
import sounddevice as sd

# ================= CONFIG =================
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

GAIN = 1.0                 # mic gain
VAD_MODE = 2               # 0–3
MIN_SPEECH_SEC = 0.8
END_SILENCE_SEC = 0.6
MAX_BUFFER_SEC = 4.0

FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
LLM_MODEL = "llama3.1:latest"
TTS_MODEL = KittenTTS("KittenML/kitten-tts-nano-0.2")

TTS_SR = 24000
TTS_VOICE = "expr-voice-5-f"
# =========================================

vad = webrtcvad.Vad(VAD_MODE)
audio_q = queue.Queue()
running = False

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

model = WhisperModel(
    "small",               # tiny / base / small
    device=device,
    compute_type=compute_type
)

# ================= AUDIO CALLBACK =================
def audio_callback(indata, frames, time_info, status):
    if not running:
        return

    # indata is now a NumPy array: shape (frames, channels)
    pcm = (indata[:, 0] * GAIN).clip(-1.0, 1.0)

    # Convert float32 → int16
    pcm_int16 = (pcm * 32767).astype(np.int16)

    audio_q.put(pcm_int16.tobytes())

# ==================ollama=========================
ollama_q = queue.Queue()

async def ask_ollama_async(user_text: str) -> str:
    response = await asyncio.to_thread(
        ollama.chat,
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a concise voice assistant. Keep responses short."
            },
            {
                "role": "user",
                "content": user_text
            }
        ],
        options={
            "temperature": 0.4,
            "num_ctx": 2048
        }
    )

    return response["message"]["content"].strip()

def ollama_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        user_text = ollama_q.get()
        if not user_text:
            continue

        reply = loop.run_until_complete(
            ask_ollama_async(user_text)
        )

        print(f"\n🤖 Kitty: {reply}")
        # send to TTS
        tts_q.put(reply)

# ================= Kitten-TTS =================
tts_q = queue.Queue()
def tts_worker():
    while True:
        text = tts_q.get()

        if not text.strip():
            continue

        try:
            audio = TTS_MODEL.generate(
                text,
                voice=TTS_VOICE
            )

            # # Play audio directly (no file needed)
            # audio = np.ascontiguousarray(audio, dtype=np.float32)
            # sd.play(audio, samplerate=TTS_SR)
            # sd.wait()  # blocking here is OK
            with sd.OutputStream(
                samplerate=TTS_SR,
                channels=1,
                dtype='float32'
                ) as stream:
                stream.write(audio)


        except Exception as e:
            print("TTS error:", e)



# ================= PROCESS THREAD =================
def processor():
    buffer = b""
    speech_time = 0.0
    speaking = False
    last_voice = time.time()

    while True:
        frame = audio_q.get()
        # buffer += frame

        is_speech = vad.is_speech(frame, SAMPLE_RATE)

        if is_speech:
            if not speaking:
                speaking = True
                buffer = b""
                speech_time = 0.0
                
            buffer += frame    # ✅ only buffer speech
            speech_time += FRAME_MS / 1000
            last_voice = time.time()

        elif speaking:
            # silence while speaking
            if time.time() - last_voice > END_SILENCE_SEC:
                speaking = False
                audio_np = (
                    np.frombuffer(buffer, np.int16)
                    .astype(np.float32) / 32768.0
                )

                segments, _ = model.transcribe(
                    audio_np,
                    language="en",          # remove for auto-detect
                    beam_size=3,
                    vad_filter=False
                )

                text = "".join(seg.text for seg in segments).strip()
                if text:
                    print(f"\n🗣️  YOU SAID: {text}")

                    ollama_q.put(text)#calling ollama function

                    # reply = ask_ollama(text) 

                    # print(f"\n🤖 OLLAMA: {reply}")

                buffer = b""
                speech_time = 0.0


# ================= CONTROL =================
def start():
    global running
    running = True
    print("\n🎙️ Listening (ENTER to stop)")


def stop():
    global running
    running = False
    print("\n⏹️ Stopped")


# ================= MAIN =================
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    blocksize=FRAME_SIZE,
    #dtype=DTYPE,
    dtype="float32",
    channels=CHANNELS,
    callback=audio_callback
)

stream.start()
threading.Thread(target=processor, daemon=True).start() # Whisper processing thread
threading.Thread(target=ollama_worker,daemon=True).start() # Ollama async worker thread
threading.Thread(target=tts_worker, daemon=True).start() # Kitten-TTS worker thread

print("Press ENTER to START / STOP | Ctrl+C to exit")

try:
    while True:
        input()
        stop() if running else start()
except KeyboardInterrupt:
    sys.exit(0)
