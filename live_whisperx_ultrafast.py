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
import gradio as gr
from faster_whisper import WhisperModel
from kittentts import KittenTTS
import sounddevice as sd

# ================= CONFIG =================
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

GAIN = 1.0                 # mic gain
VAD_MODE = 3               # 0–3
MIN_SPEECH_SEC = 1.2
END_SILENCE_SEC = 0.6
MAX_BUFFER_SEC = 4.0

FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
# LLM_MODEL = "llama3.1:latest"
LLM_MODEL = "gpt-oss:120b-cloud"

TTS_MODEL = KittenTTS("KittenML/kitten-tts-nano-0.2")

TTS_SR = 24000
TTS_VOICE = "expr-voice-5-f"
# =========================================

# =============Memory buffer===============

conversation = []
MAX_TURNS = 6  # last N exchanges

assistant_running = False
assistant_status = "Idle"

latest_transcript = ""
latest_reply = ""



# =========================================


vad = webrtcvad.Vad(VAD_MODE)
audio_q = queue.Queue()
running = False

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

model = WhisperModel(
    "medium.en",               # tiny / base / small
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

# async def ask_ollama_async(user_text: str) -> str:
async def ask_ollama_async(messages: list) -> str:

    response = await asyncio.to_thread(
        ollama.chat,
        model=LLM_MODEL,
        messages=messages,
        # messages=[
        #     {
        #         "role": "system",
        #         "content": "You are a concise voice assistant. Keep responses short."
        #     },
        #     {
        #         "role": "user",
        #         "content": user_text
        #     }
        # ],
        options={
            "temperature": 0.4,
            "num_ctx": 2048
        }
    )

    return response["message"]["content"].strip()

def ollama_worker():
    global conversation, latest_reply, assistant_status

    # ✅ create event loop ONCE, at thread start
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        user_text = ollama_q.get()

        assistant_status = "Thinking"

        # store user message
        conversation.append({"role": "user", "content": user_text})
        conversation = conversation[-MAX_TURNS * 2:]

        # build chat messages (CORRECT way)
        messages = [
            {"role": "system", "content": "You are a concise voice assistant."}
        ] + conversation

        # ✅ run async Ollama call properly
        reply = loop.run_until_complete(
            ask_ollama_async(messages)
        )

        # store assistant reply
        conversation.append({"role": "assistant", "content": reply})

        latest_reply = reply
        assistant_status = "Speaking"

        print(f"\n🤖 OLLAMA: {reply}")
        tts_q.put(reply)

# 2nd ollama worker
# def ollama_worker():
#     global conversation, latest_reply, assistant_status
#     messages = [{"role": "system", "content": "You are a concise voice assistant."}] + conversation
    
#     loop = asyncio.new_event_loop()
#     reply = loop.run_until_complete(ask_ollama_async(messages))
#     asyncio.set_event_loop(loop)

#     while True:
#         user_text = ollama_q.get()

#         conversation.append({"role": "user", "content": user_text})
#         conversation = conversation[-MAX_TURNS * 2:]

#         prompt = ""
#         for msg in conversation:
#             role = "User" if msg["role"] == "user" else "Assistant"
#             prompt += f"{role}: {msg['content']}\n"

#         prompt += "Assistant:"

#         # ✅ CORRECT: actually run the coroutine
#         reply = loop.run_until_complete(
#             ask_ollama_async(prompt)
#         )

#         conversation.append({"role": "assistant", "content": reply})

#         print(f"\n🤖 OLLAMA: {reply}")
#         latest_reply = reply
#         assistant_status = "Speaking"
#         tts_q.put(reply)



# OG ollama_worker
# def ollama_worker():
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)

#     while True:
#         user_text = ollama_q.get()
#         if not user_text:
#             continue

#         reply = loop.run_until_complete(
#             ask_ollama_async(user_text)
#         )

#         print(f"\n🤖 Kitty: {reply}")
#         # send to TTS
#         tts_q.put(reply)

# ================= Kitten-TTS =================
tts_q = queue.Queue()

def tts_worker():
    global assistant_status
    while True:
        text = tts_q.get()

        if not text or not text.strip():
            assistant_status = "Listening"
            continue

        try:
            assistant_status = "Speaking"

            audio = TTS_MODEL.generate(text, voice=TTS_VOICE)
            audio = np.asarray(audio, dtype=np.float32)

            with sd.OutputStream(
                samplerate=TTS_SR,
                channels=1,
                dtype="float32"
            ) as stream:
                stream.write(audio)

        except Exception as e:
            print("TTS error:", e)

        finally:
            assistant_status = "Listening"

# OG tts_worker
# def tts_worker():
#     global assistant_status
#     while True:
#         text = tts_q.get()

#         if not text.strip():
#             continue

#         try:
#             audio = TTS_MODEL.generate(
#                 text,
#                 voice=TTS_VOICE
#             )

#             # # Play audio directly (no file needed)
#             # audio = np.ascontiguousarray(audio, dtype=np.float32)
#             # sd.play(audio, samplerate=TTS_SR)
#             # sd.wait()  # blocking here is OK
#             with sd.OutputStream(
#                 samplerate=TTS_SR,
#                 channels=1,
#                 dtype='float32'
#                 ) as stream:
#                 stream.write(audio)
            
#             assistant_status = "Listening"



#         except Exception as e:
#             print("TTS error:", e)



# ================= PROCESS THREAD =================
def processor():
    global latest_transcript, assistant_status
    buffer = b""
    speech_time = 0.0
    speaking = False
    last_voice = time.time()

    while True:
        if not assistant_running:
            time.sleep(0.1)
            continue
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
                    latest_transcript = text
                    assistant_status = "Thinking"
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

def start_assistant():
    global assistant_running, assistant_status, running
    assistant_running = True
    running = True
    assistant_status = "Listening"
    return "Assistant started"

def stop_assistant():
    global assistant_running, assistant_status, running
    assistant_running = False
    running = False
    assistant_status = "Stopped"
    return "Assistant stopped"

def ui_refresh():
    return latest_transcript, latest_reply, assistant_status

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

# print("Press ENTER to START / STOP | Ctrl+C to exit")

# try:
#     while True:
#         input()
#         stop() if running else start()
# except KeyboardInterrupt:
#     sys.exit(0)

# ==================Gradio=================
with gr.Blocks(title="LiL-Kitty Assistant") as demo:
    gr.Markdown("## 🐱 LiL-Kitty Voice Assistant")

    with gr.Row():
        start_btn = gr.Button("🎙 Start")
        stop_btn = gr.Button("⏹ Stop")

    status = gr.Textbox(label="Status", interactive=False)

    user_box = gr.Textbox(label="You said", interactive=False)
    assistant_box = gr.Textbox(label="Assistant replied", interactive=False)

    start_btn.click(start_assistant, outputs=status)
    stop_btn.click(stop_assistant, outputs=status)

    refresh_timer = gr.Timer(0.5)
    refresh_timer.tick(
    ui_refresh,
    outputs=[user_box, assistant_box, status]
    )

#     demo.load(
#         ui_refresh,
#         outputs=[user_box, assistant_box, status],every=0.5   # refresh twice per second
#     )

demo.launch()

# =========================================