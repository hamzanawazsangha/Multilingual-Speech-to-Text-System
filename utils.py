import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os

# Load Hugging Face token from environment variable
hf_token = os.getenv("HF_TOKEN")

# Load processor and model with authentication token
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", token=hf_token)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2", token=hf_token)
model.eval()

def transcribe(audio_path):
    # Load audio
    speech_array, sampling_rate = torchaudio.load(audio_path)
    # Resample if necessary
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_array = resampler(speech_array)
    # Preprocess audio
    input_features = processor(speech_array.squeeze(), sampling_rate=16000, return_tensors="pt").input_features
    # Generate transcription
    predicted_ids = model.generate(input_features)
    # Decode transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription
