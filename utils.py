from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa

# Load model and processor once (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(device)

def transcribe(audio_path):
    speech, rate = librosa.load(audio_path, sr=16000)
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription
