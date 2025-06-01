import streamlit as st
import os
from utils import transcribe
from pydub import AudioSegment
import uuid

st.set_page_config(page_title="Multilingual STT", layout="centered")
st.title("🗣️ Multilingual Speech-to-Text System")

st.markdown("Upload an audio file (MP3/WAV) and get transcription using **Wav2Vec2**.")

uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

if uploaded_file:
    file_id = str(uuid.uuid4())
    audio_path = f"audio_files/{file_id}.wav"

    # Convert to WAV if MP3
    if uploaded_file.name.endswith(".mp3"):
        audio = AudioSegment.from_mp3(uploaded_file)
        audio.export(audio_path, format="wav")
    else:
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.read())

    st.audio(audio_path, format='audio/wav')
    st.info("Transcribing...")

    try:
        text = transcribe(audio_path)
        st.success("Transcription:")
        st.write(f"📝 {text}")
    except Exception as e:
        st.error(f"Error: {e}")
