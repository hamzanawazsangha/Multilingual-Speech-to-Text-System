import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os

# ✅ Set backend explicitly to avoid warnings
torchaudio.set_audio_backend("sox_io")

# ✅ Load Hugging Face token from environment variable (set in Streamlit Cloud secrets)
hf_token = os.getenv("HF_TOKEN")  # Or you can hardcode a token for local testing

# ✅ Load Whisper processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-small", token=hf_token)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", token=hf_token)

# ✅ Put model on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ✅ Convert language tag to ID (forcing English transcription)
forced_lang_token = processor.tokenizer.convert_tokens_to_ids("<|en|>")

def transcribe(audio_path):
    # ✅ Load and resample audio
    speech_array, sampling_rate = torchaudio.load(audio_path)
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_array = resampler(speech_array)

    input_features = processor(
        speech_array.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)

    # ✅ Generate transcription with forced language
    predicted_ids = model.generate(
        input_features,
        forced_bos_token_id=forced_lang_token
    )

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription
