import speech_recognition as sr
import numpy as np
import tensorflow as tf
import pickle
import re

# Load model
model = tf.keras.models.load_model("model/emotion_glove_model.keras")

# Load tokenizer + encoder
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

max_len = 100

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

def predict_emotion(text):
    text = clean_text(text)

    seq = tokenizer.texts_to_sequences([text])
    pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)

    pred = model.predict(pad, verbose=0)

    confidence = float(np.max(pred))
    emotion = le.inverse_transform([np.argmax(pred)])[0]

    if confidence < 0.5:
        return "neutral"

    return emotion

def listen_and_predict(retries=3):
    recognizer = sr.Recognizer()

    # IMPORTANT: improves accuracy
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8

    for attempt in range(retries):
        try:
            with sr.Microphone() as source:
                print("🎤 Listening... (attempt", attempt + 1, ")")

                # stabilize noise
                recognizer.adjust_for_ambient_noise(source, duration=1)

                # prevents hanging forever
                audio = recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=8
                )

            text = recognizer.recognize_google(audio)

            print("🗣 You said:", text)

            emotion = predict_emotion(text)
            return emotion

        except sr.WaitTimeoutError:
            print("⚠ No speech detected, retrying...")

        except sr.UnknownValueError:
            print("⚠ Could not understand, retrying...")

        except sr.RequestError:
            return "API error"

    return "Failed to capture speech after retries"