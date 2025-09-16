#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import warnings

import numpy as np
import pyaudio
import torch
import whisperx

warnings.filterwarnings("ignore")

MODEL_NAME = "small"
LANGUAGE = "ru"
COMPUTE_TYPE = "int8"
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_model():
    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        print(f"✅ Found local model: {MODEL_DIR}")
        return whisperx.load_model(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type=COMPUTE_TYPE, language=LANGUAGE, download_root=MODEL_DIR)
    else:
        print(f"📥 Downloading new model [{MODEL_NAME}] for ['{LANGUAGE}']...")
        print("Downloading.... (few minutes)")
        try:
            model = whisperx.load_model(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu",
                compute_type=COMPUTE_TYPE, language=LANGUAGE, download_root=MODEL_DIR)
            print(f"💾 Model saved to: {MODEL_DIR}")
            return model
        except Exception as e:
            print(f" Cant make model: {e}")
            sys.exit(1)


model = load_model()

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SEGMENT_DURATION = 5
SEGMENT_SAMPLES = RATE * SEGMENT_DURATION

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print(f"🎙️ Live started for {MODEL_NAME}, lang: {LANGUAGE}")
print("Speak:...")

buffer = np.array([], dtype=np.float32)

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        buffer = np.concatenate([buffer, audio_chunk])

        if len(buffer) >= SEGMENT_SAMPLES:
            result = model.transcribe(buffer[:SEGMENT_SAMPLES], batch_size=16)
            for seg in result["segments"]:
                text = seg["text"].strip()
                if text:
                    print(text)
            buffer = buffer[SEGMENT_SAMPLES:]

except KeyboardInterrupt:
    print("\nStopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()
