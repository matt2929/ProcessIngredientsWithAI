import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import soundfile as sf
import logging
import time
from tqdm import tqdm
import librosa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Load model & processor
logging.info("Loading model and processor...")
model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

# Load audio file
filename = "/Users/matthewstafford/Desktop/video_audio.wav"
logging.info(f"Loading audio file: {filename}")
speech, sr = sf.read(filename)

# Convert to mono if stereo
if speech.ndim > 1:
    logging.info("Converting stereo to mono...")
    speech = speech.mean(axis=1)

# Chunking parameters
chunk_duration = 20  # seconds
chunk_size = sr * chunk_duration
chunks = [speech[i:i + chunk_size] for i in range(0, len(speech), chunk_size)]
logging.info(f"Audio split into {len(chunks)} chunks of {chunk_duration} seconds each (at {sr} Hz)")

# Process each chunk with tqdm progress bar
translations = []
for idx, chunk in enumerate(tqdm(chunks, desc="Processing chunks", unit="chunk")):
    if len(chunk) == 0:
        continue

    start_time = time.time()
    logging.info(f"Processing chunk {idx + 1}/{len(chunks)}")

    # Resample THIS chunk only (if needed)
    if sr != 16000:
        chunk = librosa.resample(chunk, orig_sr=sr, target_sr=16000)
        chunk_sr = 16000
    else:
        chunk_sr = sr

    # Prepare inputs
    inputs = processor(chunk, sampling_rate=chunk_sr, return_tensors="pt")

    # Generate translation
    generated_ids = model.generate(inputs["input_features"])
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    elapsed = time.time() - start_time
    logging.info(f"Chunk {idx + 1} finished in {elapsed:.2f}s: {text}")

    translations.append(text)

# Join all results
full_translation = " ".join(translations)
logging.info("=== Final Translation ===")
print(full_translation)
