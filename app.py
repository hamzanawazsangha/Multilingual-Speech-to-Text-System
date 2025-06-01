import streamlit as st
import os
import uuid
from pydub import AudioSegment
from utils import transcribe

# Set Streamlit page configuration
st.set_page_config(page_title="🗣️ Multilingual STT", layout="centered")
st.title("🗣️ Multilingual Speech-to-Text System")

st.markdown("Upload an audio file (MP3/WAV) and get transcription using **Whisper**.")

# File uploader
uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

if uploaded_file:
    # Create a unique filename
    file_id = str(uuid.uuid4())
    audio_dir = "audio_files"
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, f"{file_id}.wav")

    # Convert to WAV if MP3
    if uploaded_file.name.endswith(".mp3"):
        audio = AudioSegment.from_file(uploaded_file, format="mp3")
        audio.export(audio_path, format="wav")
    else:
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.read())

    # Display audio player
    st.audio(audio_path, format='audio/wav')
    st.info("Transcribing...")

    # Transcribe audio
    try:
        text = transcribe(audio_path)
        st.success("Transcription:")
        st.write(f"📝 {text}")
    except Exception as e:
        st.error(f"Error: {e}")
