from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
import os

# Hugging Face token
hf_token = os.getenv("HF_TOKEN", None)

# Load Whisper model (multilingual)
processor = WhisperProcessor.from_pretrained("openai/whisper-large", use_auth_token=hf_token)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large", use_auth_token=hf_token)

# Force decoding to English (can be customized)
forced_lang_token = processor.tokenizer.lang_to_id.get("en", None)

def transcribe(audio_path):
    speech_array, sampling_rate = torchaudio.load(audio_path)
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        speech_array = resampler(speech_array)

    input_features = processor(speech_array.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features
    generated_ids = model.generate(input_features, forced_bos_token_id=forced_lang_token)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription
