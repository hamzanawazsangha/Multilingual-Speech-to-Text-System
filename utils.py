import torch
import torchaudio
from transformers import pipeline
import os
from typing import Optional

# Initialize the pipeline (simpler interface)
def get_model():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",  # Using medium instead of large for better performance
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

# Global model loading (cached)
pipe = get_model()

def transcribe(audio_path: str) -> Optional[str]:
    """
    Transcribe audio file to text using Whisper model
    
    Args:
        audio_path: Path to audio file (WAV format recommended)
        
    Returns:
        Transcribed text or None if error occurs
    """
    try:
        # Use the pipeline which handles all preprocessing automatically
        result = pipe(
            audio_path,
            chunk_length_s=30,  # For longer audio files
            stride_length_s=5,
            batch_size=8
        )
        return result["text"]
        
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        raise
