import torch
from transformers import pipeline
import os

# Initialize the pipeline with proper settings
def get_model():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

# Global model loading with cache
pipe = get_model()

def transcribe(audio_path: str, language: str = None) -> str:
    """Transcribe audio with optional language specification"""
    try:
        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language
            
        result = pipe(
            audio_path,
            chunk_length_s=30,
            stride_length_s=5,
            batch_size=4,
            generate_kwargs=generate_kwargs
        )
        return result["text"]
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")
