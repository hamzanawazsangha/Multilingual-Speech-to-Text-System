import streamlit as st
import os
import uuid
from pydub import AudioSegment
from utils import transcribe

# Page setup
st.set_page_config(page_title="🗣️ Multilingual STT", layout="centered")
st.title("🗣️ Multilingual Speech-to-Text")
st.markdown("Upload audio in **any language** and get fast, accurate transcription using OpenAI Whisper!")

# Ensure audio directory exists
os.makedirs("audio_files", exist_ok=True)

# App tabs
tab1, tab2 = st.tabs(["🎧 Transcribe", "ℹ️ About"])

with tab1:
    uploaded_file = st.file_uploader("Upload MP3 or WAV file", type=["mp3", "wav"])
    
    if uploaded_file:
        file_id = str(uuid.uuid4())
        audio_path = f"audio_files/{file_id}.wav"

        try:
            if uploaded_file.name.endswith(".mp3"):
                audio = AudioSegment.from_file(uploaded_file, format="mp3")
                audio.export(audio_path, format="wav")
            else:
                with open(audio_path, "wb") as f:
                    f.write(uploaded_file.read())

            st.audio(audio_path, format="audio/wav")
            
            with st.spinner("Transcribing... Please wait."):
                transcription = transcribe(audio_path)
                st.success("✅ Transcription completed!")
                st.text_area("📝 Transcribed Text:", transcription, height=200)

        except Exception as e:
            st.error(f"❌ Error: {e}")

with tab2:
    st.subheader("About the App")
    st.markdown("""
    This multilingual speech-to-text application is powered by **OpenAI Whisper**, 
    which supports a wide range of global languages. The system transcribes uploaded audio files into written text.

    **Key Features:**
    - Supports multiple languages
    - Works with MP3 and WAV files
    - Powered by Hugging Face and OpenAI Whisper

    _Built with ❤️ using Streamlit_
    """)
