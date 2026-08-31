import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time

from response_engine import get_aiva_response


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AIVA Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    gender_model = tf.keras.models.load_model(
        "model/gender_model.keras",
        compile=False
    )

    age_model = tf.keras.models.load_model(
        "model/age_model.keras",
        compile=False
    )

    return gender_model, age_model


try:
    gender_model, age_model = load_models()
except Exception as e:
    st.error("❌ Failed to load AI models.")
    st.exception(e)
    st.stop()


# ============================================================
# AGE LABELS
# ============================================================

age_groups = [
    "Young",
    "Adult",
    "Old"
]

age_ranges = [
    "0-17 yr",
    "18-39 yr",
    "40+ yr"
]


# ============================================================
# FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# HEADER
# ============================================================

st.title("🤖 AIVA - AI Image & Voice Analyzer")

st.write(
    "Use your browser camera below to allow AIVA to analyze the image."
)


# ============================================================
# CAMERA
# ============================================================

camera_image = st.camera_input(
    "📷 Take a photo"
)


# ============================================================
# ANALYZE CAMERA IMAGE
# ============================================================

label_text = "No face detected"

if camera_image is not None:

    # Read uploaded camera image
    image_bytes = camera_image.getvalue()

    np_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        st.error("❌ Could not read camera image.")
        st.stop()

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # --------------------------------------------------------
    # Analyze faces
    # --------------------------------------------------------

    for (x, y, w, h) in faces:

        # Crop face
        face = frame[
            y:y + h,
            x:x + w
        ]

        if face.size == 0:
            continue

        # Resize to model input
        face = cv2.resize(
            face,
            (64, 64)
        )

        # Normalize
        face = face.astype(np.float32) / 255.0

        # Add batch dimension
        face = np.expand_dims(
            face,
            axis=0
        )

        # ----------------------------------------------------
        # Gender prediction
        # ----------------------------------------------------

        gender_pred = gender_model.predict(
            face,
            verbose=0
        )

        gender_score = float(
            np.asarray(gender_pred).reshape(-1)[0]
        )

        gender = (
            "Female"
            if gender_score > 0.5
            else "Male"
        )

        # ----------------------------------------------------
        # Age prediction
        # ----------------------------------------------------

        age_pred = age_model.predict(
            face,
            verbose=0
        )

        age_pred = np.asarray(age_pred)

        age_class = int(
            np.argmax(age_pred)
        )

        # Protect against invalid model output
        if age_class >= len(age_groups):
            age_class = 0

        age_group = age_groups[age_class]
        age_range = age_ranges[age_class]

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label_text = (
            f"{gender} | "
            f"{age_group} "
            f"({age_range})"
        )

        # ----------------------------------------------------
        # Draw face box
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Make sure text doesn't go outside image
        text_y = max(
            y - 10,
            25
        )

        cv2.putText(
            frame,
            label_text,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        frame_rgb,
        caption="AIVA Camera Analysis",
        use_container_width=True
    )


# ============================================================
# DETECTION INFO
# ============================================================

st.markdown("---")

info_box = st.empty()

info_box.markdown(
    f"""
    ### 🧑 Detection

    **Gender / Age:** {label_text}
    """
)


# ============================================================
# VOICE SECTION
# ============================================================

st.markdown("---")

st.subheader("🎤 Voice Emotion")

st.info(
    "Browser microphone access cannot be handled by PyAudio "
    "running on the Render server. Use a browser audio component "
    "for live microphone input."
)


# ============================================================
# AIVA RESPONSE
# ============================================================

emotion_text = "Waiting for voice input..."

try:
    aiva_response = get_aiva_response(
        emotion_text
    )
except Exception:
    aiva_response = "Waiting for voice input..."


st.markdown(
    f"""
    ### 🎤 Voice Emotion

    **{emotion_text}**
    """
)

st.markdown(
    f"""
    ### 🤖 AIVA Response

    **{aiva_response}**
    """
)
