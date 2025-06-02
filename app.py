import streamlit as st
import os
import uuid
from pydub import AudioSegment
from utils import transcribe

# Workaround for Streamlit/PyTorch compatibility
os.environ['STREAMLIT_PYTORCH_PATH_WORKAROUND'] = '1'

# Set Streamlit page configuration
st.set_page_config(
    page_title="🗣️ Multilingual STT",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🗣️ Multilingual Speech-to-Text System")

# Language selection
language = st.selectbox(
    "Select Language (or leave as 'auto' for detection)",
    ["auto", "en", "es", "fr", "de", "ja", "zh", "hi", "ar"],
    index=0
)

uploaded_file = st.file_uploader(
    "Upload Audio (MP3/WAV/OGG)",
    type=["wav", "mp3", "ogg"],
    accept_multiple_files=False
)

if uploaded_file:
    file_id = str(uuid.uuid4())
    audio_dir = "temp_audio"
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, f"{file_id}.wav")

    try:
        # Audio processing with error handling
        audio = AudioSegment.from_file(uploaded_file)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(audio_path, format="wav")

        st.audio(audio_path, format='audio/wav')
        
        with st.spinner("Transcribing... This may take a few moments..."):
            text = transcribe(audio_path, language=language if language != "auto" else None)
            
        st.success("Transcription Complete!")
        st.text_area("Result", value=text, height=150)
        
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
