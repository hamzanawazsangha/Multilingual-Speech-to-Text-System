import streamlit as st
import os
import uuid
from pydub import AudioSegment
from utils import transcribe

# Set Streamlit page configuration
st.set_page_config(page_title="🗣️ Multilingual STT", layout="centered")
st.title("🗣️ Multilingual Speech-to-Text System")

st.markdown("""
Upload an audio file (MP3/WAV) and get transcription using **Whisper**.
- For best results, use clear audio with minimal background noise
- Ideal audio length is under 30 seconds for quick processing
- Supports multiple languages automatically
""")

# File uploader
uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg"])

if uploaded_file:
    # Create a unique filename
    file_id = str(uuid.uuid4())
    audio_dir = "audio_files"
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, f"{file_id}.wav")

    try:
        # Convert to WAV if needed and ensure mono channel
        if uploaded_file.name.endswith(".mp3"):
            audio = AudioSegment.from_file(uploaded_file, format="mp3")
        else:
            audio = AudioSegment.from_file(uploaded_file)
        
        # Convert to mono and set frame rate
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(audio_path, format="wav")

        # Display audio player
        st.audio(audio_path, format='audio/wav')
        
        with st.spinner("Transcribing audio... This may take a moment..."):
            # Transcribe audio
            text = transcribe(audio_path)
            st.success("Transcription Complete!")
            st.text_area("Transcription:", value=text, height=150)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try a different audio file or check the file format.")
    finally:
        # Clean up audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
