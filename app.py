import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import keras

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
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():

    gender_model = keras.models.load_model(
        "model/gender_model.keras",
        compile=False,
        safe_mode=False
    )

    age_model = keras.models.load_model(
        "model/age_model.keras",
        compile=False,
        safe_mode=False
    )

    return gender_model, age_model


# ============================================================
# LOAD AI MODELS
# ============================================================

try:

    gender_model, age_model = load_models()

except Exception as e:

    st.error("❌ Failed to load AI models.")

    st.code(
        str(e),
        language="text"
    )

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

if face_cascade.empty():

    st.error("❌ OpenCV face detector could not be loaded.")

    st.stop()


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
# DEFAULT RESULT
# ============================================================

label_text = "No face detected"


# ============================================================
# CAMERA ANALYSIS
# ============================================================

if camera_image is not None:

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

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
    # Grayscale
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
    # No face
    # --------------------------------------------------------

    if len(faces) == 0:

        label_text = "No face detected"


    # --------------------------------------------------------
    # Analyze faces
    # --------------------------------------------------------

    for (x, y, w, h) in faces:

        face = frame[
            y:y + h,
            x:x + w
        ]

        if face.size == 0:
            continue


        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        face = cv2.resize(
            face,
            (64, 64)
        )


        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        face = face.astype(
            np.float32
        ) / 255.0


        # ----------------------------------------------------
        # Batch dimension
        # ----------------------------------------------------

        face = np.expand_dims(
            face,
            axis=0
        )


        # ====================================================
        # GENDER
        # ====================================================

        gender_pred = gender_model.predict(
            face,
            verbose=0
        )

        gender_score = float(
            np.asarray(
                gender_pred
            ).reshape(-1)[0]
        )


        gender = (
            "Female"
            if gender_score > 0.5
            else "Male"
        )


        # ====================================================
        # AGE
        # ====================================================

        age_pred = age_model.predict(
            face,
            verbose=0
        )

        age_pred = np.asarray(
            age_pred
        )


        age_class = int(
            np.argmax(age_pred)
        )


        # Protect against unexpected output
        if age_class < 0 or age_class >= len(age_groups):

            age_class = 0


        age_group = age_groups[
            age_class
        ]

        age_range = age_ranges[
            age_class
        ]


        # ====================================================
        # RESULT LABEL
        # ====================================================

        label_text = (
            f"{gender} | "
            f"{age_group} "
            f"({age_range})"
        )


        # ====================================================
        # DRAW FACE BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # ----------------------------------------------------
        # Text position
        # ----------------------------------------------------

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
    # DISPLAY IMAGE
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
# DETECTION INFORMATION
# ============================================================

st.markdown("---")

st.subheader("🧑 Detection")

st.write(
    f"**Gender / Age:** {label_text}"
)


# ============================================================
# VOICE
# ============================================================

st.markdown("---")

st.subheader("🎤 Voice Emotion")

st.info(
    "Browser microphone input will be connected separately. "
    "The Render server cannot directly access your computer's microphone."
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

    aiva_response = (
        "Waiting for voice input..."
    )


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
