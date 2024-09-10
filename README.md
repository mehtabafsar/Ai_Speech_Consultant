# AI Speech Processing Toolkit

Welcome to the AI and Speech Processing Toolkit! This project combines state-of-the-art speech-to-text, text generation, and audio recording functionalities to enhance your interaction with audio data.

## 🚀 Features

- **Speech-to-Text**: Convert spoken language into written text using the Wav2Vec2 model.
- **Text Generation**: Generate passages of text using GPT-2 based on input prompts.
- **Audio Recording**: Record and save audio files for further processing.

## 🛠️ Installation

To get started, you'll need to install the necessary libraries. Run the following command:

```bash
pip install transformers datasets librosa torch sounddevice wavio huggingface_hub
```

## 📋 Usage

### 1. Speech-to-Text

This feature transcribes audio files into text. 

```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import torch

# Load the pretrained model and processor
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Preprocess the audio data
def preprocess_function(examples):
    speech_array, sampling_rate = librosa.load(examples["path"], sr=16000)
    inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return {"transcription": transcription}
```

### 2. Text Generation

Generate text passages using GPT-2.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_passage(prompt="Please speak about the importance of effective communication."):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)
    passage = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return passage
```

### 3. Audio Recording

Record and save audio files.

```python
import sounddevice as sd
import wavio

def record_audio(filename, duration=10, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
    sd.wait()  # Wait until recording is finished
    wavio.write(filename, recording, fs, sampwidth=2)
    print("Recording saved as", filename)
```

## 📚 Examples

1. **Generate a Passage**:

```python
passage = generate_passage()
print("Generated Passage:")
print(passage)
```

2. **Record Audio**:

```python
record_audio("user_speech.wav")
```



For any questions or collaborations, feel free to reach out!

