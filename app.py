import os
import time
import torch
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline
import gc

# Set page config
st.set_page_config(
    page_title="Multilingual Speech-to-Text (CPU Optimized)",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def load_css():
    st.markdown("""
    <style>
        /* Simplified styling for better CPU performance */
        .stApp {
            background-color: #f0f2f6;
        }
        .title {
            font-size: 2em;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 0.5em;
        }
        .card {
            background-color: white;
            border-radius: 8px;
            padding: 1.5em;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 1.5em;
        }
        .stButton>button {
            background-color: #4a6fa5;
            color: white;
            border-radius: 4px;
            padding: 0.5em 1em;
        }
        .result-box {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 1em;
            border-left: 4px solid #4a6fa5;
        }
        /* Disable animations for CPU */
        * {
            transition: none !important;
            animation: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model(model_name="openai/whisper-base"):
    """Load a smaller model optimized for CPU"""
    try:
        # Force CPU usage
        device = "cpu"
        
        # Use simpler model configuration for CPU
        model = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            torch_dtype=torch.float32  # Use float32 on CPU
        )
        
        # Reduce model footprint
        if hasattr(model.model, "encoder"):
            model.model.encoder.layer.drop = 0.0
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def save_audio_file(audio_bytes, file_extension):
    """Save audio bytes to a file with the specified extension"""
    timestamp = int(time.time())
    file_name = f"audio_{timestamp}.{file_extension}"
    
    with open(file_name, "wb") as f:
        f.write(audio_bytes)
    
    return file_name

def transcribe_audio(model, audio_file_path, language):
    """Transcribe the audio file using the model with CPU optimizations"""
    try:
        # Show warning for large models
        if "large" in model.model.name_or_path:
            st.warning("Large model selected - this may be slow on CPU")
        
        # Set simpler generation config for CPU
        generate_kwargs = {
            "task": "transcribe",
            "language": language if language != "auto" else None,
            "without_timestamps": True  # Simpler output for CPU
        }
        
        # Transcribe in smaller chunks
        with st.spinner("Transcribing (this may take a while on CPU)..."):
            result = model(
                audio_file_path,
                generate_kwargs=generate_kwargs,
                chunk_length_s=20,  # Smaller chunks for CPU
                batch_size=4  # Smaller batch for CPU
            )
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result["text"]
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

def main():
    load_css()
    
    # Title and description
    st.markdown('<h1 class="title">🎙️ CPU-Optimized Speech-to-Text</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Lightweight version for CPU-only environments</p>', unsafe_allow_html=True)
    
    # Sidebar with simplified options
    with st.sidebar:
        st.markdown("## Settings")
        
        # Model selection - only smaller models
        model_size = st.selectbox(
            "Model Size (recommend: base)",
            ["tiny", "base", "small"],
            index=1,
            help="Larger models are very slow on CPU"
        )
        
        # Limited language selection
        language = st.selectbox(
            "Spoken Language",
            ["auto", "english", "spanish", "french", "german"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### Tips for CPU")
        st.markdown("- Use short audio clips (<30 seconds)")
        st.markdown("- Stick with 'base' or 'tiny' models")
        st.markdown("- Close other browser tabs")
    
    # Main content
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Audio recorder with simpler interface
        st.markdown("### Record Your Voice")
        audio_bytes = audio_recorder(
            pause_threshold=10.0,
            sample_rate=16_000,
            text="Click to record (max 10s)",
            neutral_color="#6aa36f",
            energy_threshold=(-1.0, 1.0)
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            audio_file_path = save_audio_file(audio_bytes, "wav")
            
            if st.button("Transcribe Audio", key="transcribe"):
                with st.spinner("Loading model (first time may take a while)..."):
                    model = load_model(f"openai/whisper-{model_size}")
                
                if model:
                    transcription = transcribe_audio(model, audio_file_path, language)
                    
                    if transcription:
                        st.markdown("### Transcription Result")
                        st.markdown(f'<div class="result-box">{transcription}</div>', unsafe_allow_html=True)
                        
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
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
