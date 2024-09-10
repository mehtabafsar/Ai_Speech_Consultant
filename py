# Install necessary libraries
!pip install transformers datasets librosa torch sounddevice wavio huggingface_hub

# Import necessary libraries
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, GPT2LMHeadModel, GPT2Tokenizer
import librosa
import torch
from datasets import load_dataset
from huggingface_hub import login
import sounddevice as sd
import wavio

# Load the pretrained model and processor for speech-to-text
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Authenticate with Hugging Face
login(token='your_token_here')  # Replace with your actual token

# Load dataset
dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train")

# Preprocess the audio data
def preprocess_function(examples):
    try:
        # Assuming `examples` is a dictionary with 'path' field
        speech_array, sampling_rate = librosa.load(examples["path"], sr=16000)
        inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_values = inputs.input_values
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return {"transcription": transcription}
    except Exception as e:
        print(f"Error processing example: {e}")
        return {"transcription": "Error"}

# Apply the preprocessing function to the dataset
processed_dataset = dataset.map(preprocess_function, batched=True)

# Function to analyze speech and provide feedback
def analyze_speech(audio_file):
    try:
        # Load and preprocess the audio
        speech_array, sampling_rate = librosa.load(audio_file, sr=16000)
        inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_values = inputs.input_values
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        # Analyze the transcription for areas of improvement
        feedback = []
        if "uh" in transcription or "um" in transcription:
            feedback.append("Consider reducing filler words like 'uh' and 'um'.")
        # Add more sophisticated analysis for vocabulary, accent, etc.

        return transcription, feedback
    except Exception as e:
        print(f"Error analyzing speech: {e}")
        return "Error", []

# Generate passages using GPT-2
def generate_passage(prompt="Please speak about the importance of effective communication."):
    try:
        model_name = "gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(inputs, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)
        passage = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return passage
    except Exception as e:
        print(f"Error generating passage: {e}")
        return "Error"

# Record audio for the passage
def record_audio(filename, duration=10, fs=44100):
    try:
        print("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
        sd.wait()  # Wait until recording is finished
        wavio.write(filename, recording, fs, sampwidth=2)
        print("Recording saved as", filename)
    except Exception as e:
        print(f"Error recording audio: {e}")

# Example usage
# Generate a passage
passage = generate_passage()
print("Generated Passage:")
print(passage)

# Record audio for the passage
record_audio("user_speech.wav")
