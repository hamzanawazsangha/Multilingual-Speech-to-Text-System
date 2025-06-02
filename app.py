import os
import time
import torch
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

# Set page config
st.set_page_config(
    page_title="Multilingual Speech-to-Text",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def load_css():
    st.markdown("""
    <style>
        /* Main styles */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .title {
            font-size: 2.5em;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 0.5em;
        }
        .subtitle {
            font-size: 1.2em;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 2em;
        }
        /* Card styles */
        .card {
            background-color: white;
            border-radius: 10px;
            padding: 2em;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 2em;
        }
        /* Button styles */
        .stButton>button {
            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
            color: white;
            border: none;
            border-radius: 50px;
            padding: 0.5em 1.5em;
            font-size: 1em;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        /* Audio recorder */
        .audio-recorder {
            margin: 0 auto;
            width: 100%;
        }
        /* Progress bar */
        .stProgress > div > div > div {
            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        }
        /* Sidebar */
        .sidebar .sidebar-content {
            background: linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%);
            color: white;
        }
        /* Language select */
        .stSelectbox > div > div {
            background-color: white;
            border-radius: 10px;
        }
        /* Result box */
        .result-box {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 1.5em;
            border-left: 5px solid #6a11cb;
            margin-top: 1em;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model(model_name="openai/whisper-medium"):
    """Load the Whisper model with optimizations"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Use flash attention if available
    attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
    
    model = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
        torch_dtype=torch_dtype,
        model_kwargs={"attn_implementation": attn_implementation}
    )
    return model

def save_audio_file(audio_bytes, file_extension):
    """Save audio bytes to a file with the specified extension"""
    timestamp = int(time.time())
    file_name = f"audio_{timestamp}.{file_extension}"
    
    with open(file_name, "wb") as f:
        f.write(audio_bytes)
    
    return file_name

def transcribe_audio(model, audio_file_path, language):
    """Transcribe the audio file using the model"""
    try:
        # Set generation config based on language
        generate_kwargs = {"task": "transcribe"}
        if language != "auto":
            generate_kwargs["language"] = language
        
        # Transcribe with progress
        with st.spinner("Transcribing audio..."):
            result = model(
                audio_file_path,
                generate_kwargs=generate_kwargs,
                chunk_length_s=30,
                batch_size=8
            )
        
        return result["text"]
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

def main():
    load_css()
    
    # Title and description
    st.markdown('<h1 class="title">🎙️ Multilingual Speech-to-Text</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Convert speech to text in multiple languages using AI</p>', unsafe_allow_html=True)
    
    # Sidebar with options
    with st.sidebar:
        st.markdown("## Settings")
        
        # Model selection
        model_size = st.selectbox(
            "Model Size",
            ["tiny", "base", "small", "medium", "large"],
            index=3,
            help="Larger models are more accurate but slower"
        )
        
        # Language selection
        language = st.selectbox(
            "Spoken Language",
            [
                "auto", "english", "spanish", "french", "german", 
                "italian", "portuguese", "dutch", "russian", 
                "japanese", "chinese", "arabic", "hindi"
            ],
            index=0,
            help="Select 'auto' for automatic language detection"
        )
        
        # Audio source selection
        audio_source = st.radio(
            "Audio Source",
            ["Record Audio", "Upload Audio File"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This app uses OpenAI's Whisper model through HuggingFace Transformers to convert speech to text in multiple languages.")
        st.markdown("**Note:** The first run may take a minute to download the model.")
    
    # Main content
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Audio input based on selection
        audio_file_path = None
        
        if audio_source == "Record Audio":
            st.markdown("### Record Your Voice")
            audio_bytes = audio_recorder(
                pause_threshold=5.0,
                sample_rate=16_000,
                text="Click to start recording",
                recording_color="#e8b62c",
                neutral_color="#6aa36f",
                icon_name="microphone",
                icon_size="2x",
            )
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                audio_file_path = save_audio_file(audio_bytes, "wav")
        
        else:  # Upload Audio File
            st.markdown("### Upload Audio File")
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=["wav", "mp3", "ogg", "flac"]
            )
            
            if uploaded_file:
                file_extension = uploaded_file.name.split(".")[-1]
                audio_bytes = uploaded_file.read()
                st.audio(audio_bytes, format=f"audio/{file_extension}")
                audio_file_path = save_audio_file(audio_bytes, file_extension)
        
        # Transcribe button
        if audio_file_path and st.button("Transcribe Audio", use_container_width=True):
            model = load_model(f"openai/whisper-{model_size}")
            transcription = transcribe_audio(model, audio_file_path, language)
            
            if transcription:
                st.markdown("### Transcription Result")
                st.markdown(f'<div class="result-box">{transcription}</div>', unsafe_allow_html=True)
                
                # Add copy button
                st.download_button(
                    label="Copy to Clipboard",
                    data=transcription,
                    file_name="transcription.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # Clean up the audio file
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
