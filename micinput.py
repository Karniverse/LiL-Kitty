import gradio as gr
import numpy as np

THRESHOLD = 0.01
SILENCE_FRAMES = 20
silence_counter = 0

def extract_pcm(audio_chunk):
    """Return numpy float32 mono PCM from any Gradio streaming format."""
    # Case 1: Dict format
    if isinstance(audio_chunk, dict):
        data = audio_chunk.get("data")
        pcm = np.array(data, dtype=np.float32)

    # Case 2: Tuple format (pcm, sample_rate)
    elif isinstance(audio_chunk, tuple) and len(audio_chunk) == 2:
        pcm, sr = audio_chunk
        pcm = np.array(pcm, dtype=np.float32)

    # Cannot process
    else:
        return None

    # Flatten stereo → mono
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=0)

    return pcm


def stream_audio(audio_chunk):
    global silence_counter

    pcm = extract_pcm(audio_chunk)
    if pcm is None:
        return None, "Waiting..."

    # Compute volume (simple RMS)
    volume = np.abs(pcm).mean()

    # Silence detection
    if volume < THRESHOLD:
        silence_counter += 1
    else:
        silence_counter = 0

    if silence_counter > SILENCE_FRAMES:
        return None, "Auto-stopped (silence)"

    return None, f"Volume: {volume:.5f}"


with gr.Blocks() as demo:
    audio = gr.Audio(sources=["microphone"], streaming=True)
    status = gr.Textbox()

    audio.stream(stream_audio, audio, status)

demo.launch()
