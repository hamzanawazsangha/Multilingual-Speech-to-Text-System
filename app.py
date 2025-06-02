import os
import time
import torch
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline
import gc

# Set page config
st.set_page_config(
    page_title="Multilingual Speech-to-Text",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="auto"
)

# Simplified CSS for better compatibility
def load_css():
    st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .title {
            font-size: 1.8em;
            color: #2c3e50;
            text-align: center;
        }
        .card {
            background-color: white;
            border-radius: 8px;
            padding: 1em;
            margin: 1em 0;
        }
        .stButton>button {
            background-color: #4a6fa5;
            color: white;
        }
        .tab-content {
            padding: 1em 0;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model(model_name="openai/whisper-tiny"):
    """Load a CPU-optimized model"""
    try:
        model = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device="cpu",
            torch_dtype=torch.float32
        )
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def save_audio_file(audio_bytes, file_extension):
    """Save audio bytes to a file"""
    timestamp = int(time.time())
    file_name = f"audio_{timestamp}.{file_extension}"
    with open(file_name, "wb") as f:
        f.write(audio_bytes)
    return file_name

def transcribe_audio(model, audio_file_path, language):
    """Transcribe audio with CPU optimizations"""
    try:
        generate_kwargs = {
            "task": "transcribe",
            "language": language if language != "auto" else None
        }
        
        with st.spinner("Processing audio..."):
            result = model(
                audio_file_path,
                generate_kwargs=generate_kwargs,
                chunk_length_s=15,
                batch_size=2
            )
        
        gc.collect()
        return result["text"]
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

def main():
    load_css()
    
    st.title("🎙️ Multilingual Speech-to-Text")
    st.caption("Lightweight version optimized for CPU")

    with st.sidebar:
        st.subheader("Settings")
        model_size = st.selectbox(
            "Model Size",
            ["tiny", "base"],
            index=0,
            help="'tiny' is fastest on CPU"
        )
        language = st.selectbox(
            "Language",
            ["auto", "english", "spanish", "french", "german"],
            index=0
        )

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Record Audio", "Upload Audio File"])

    audio_file_path = None
    
    with tab1:
        st.subheader("Record Your Voice")
        audio_bytes = audio_recorder(
            pause_threshold=10.0,
            sample_rate=16_000,
            text="Click to record (max 10s)",
            neutral_color="#6aa36f"
        )

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            audio_file_path = save_audio_file(audio_bytes, "wav")

    with tab2:
        st.subheader("Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "ogg"],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            file_extension = uploaded_file.name.split(".")[-1]
            st.audio(uploaded_file, format=f"audio/{file_extension}")
            audio_file_path = save_audio_file(uploaded_file.read(), file_extension)

    # Transcribe button (appears when audio is available)
    if audio_file_path and st.button("Transcribe Audio"):
        model = load_model(f"openai/whisper-{model_size}")
        
        if model:
            transcription = transcribe_audio(model, audio_file_path, language)
            if transcription:
                st.text_area("Transcription", transcription, height=150)
                
                # Add download button
                st.download_button(
                    label="Download Transcription",
                    data=transcription,
                    file_name="transcription.txt",
                    mime="text/plain"
                )
        
        # Clean up
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
        gc.collect()

if __name__ == "__main__":
    main()
