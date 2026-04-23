import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import threading
import time

from audio_emotion import listen_and_predict
from response_engine import get_aiva_response


gender_model = tf.keras.models.load_model("model/gender_model.keras")
age_model = tf.keras.models.load_model("model/age_model.keras")

age_groups = ["Young", "Adult", "Old"]
age_ranges = ["0-17 yr", "18-39 yr", "40+ yr"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


st.set_page_config(page_title="AIVA Assistant", layout="wide")

st.title("🤖 AIVA - AI Virtual Assistant")

# placeholders (IMPORTANT for smooth updates)
frame_window = st.image([])
info_box = st.empty()
emotion_box = st.empty()
response_box = st.empty()


emotion_text = "Listening..."
aiva_response = "Waiting..."
running = True


def audio_loop():
    global emotion_text, aiva_response

    while running:
        try:
            emotion_text = listen_and_predict()
            aiva_response = get_aiva_response(emotion_text)
        except:
            emotion_text = "Audio error"
            aiva_response = "..."

        time.sleep(2)  # prevents CPU overload


threading.Thread(target=audio_loop, daemon=True).start()



cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Camera not accessible")
    st.stop()


while True:

    ret, frame = cap.read()
    if not ret:
        st.warning("Camera frame not received")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    label_text = "No face detected"

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = np.reshape(face, (1, 64, 64, 3))

        # Gender
        gender_pred = gender_model.predict(face, verbose=0)
        gender = "Female" if gender_pred[0][0] > 0.5 else "Male"

        # Age
        age_pred = age_model.predict(face, verbose=0)
        age_class = np.argmax(age_pred)

        age_group = age_groups[age_class]
        age_range = age_ranges[age_class]

        label_text = f"{gender} | {age_group} ({age_range})"

        # Draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

    # Convert BGR → RGB (Streamlit requirement)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_window.image(frame, channels="RGB")

    info_box.markdown(f"""
    ### 🧑 Detection
    - **Gender/Age:** {label_text}
    """)

    emotion_box.markdown(f"""
    ### 🎤 Voice Emotion
    **{emotion_text}**
    """)

    response_box.markdown(f"""
    ### 🤖 AIVA Response
    **{aiva_response}**
    """)

    time.sleep(0.03)  # smooth UI

# cleanup
cap.release()