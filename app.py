import gradio as gr
import librosa
from transformers import pipeline

# loading ASR model
asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

def transcribe(audio_file):
    audio, rate = librosa.load(audio_file, sr=16000)
    res = asr(audio)
    return res["text"]

# Gradio UI
interface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Upload a short WAV audio"),
    outputs="text",
    title="Speech-to-Text with Wav2Vec2",
    description="For best performance, upload .wav audio clips under 30 seconds."
)

interface.launch()   
