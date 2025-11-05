# ----------------------------------------------------------
# Emotion Detection Streamlit App (FER2013 model)
# ----------------------------------------------------------

import io
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Try to import MTCNN for better face detection
USE_MTCNN = True
try:
    from mtcnn import MTCNN
except Exception:
    USE_MTCNN = False

# ----------------------------------------------------------
# Page setup
# ----------------------------------------------------------
st.set_page_config(page_title="😄 Emotion Detector", layout="centered")
st.title("😄 Emotion Detection App")
st.write("Upload a face image to predict the emotion using your **FER2013 model**.")

# ----------------------------------------------------------
# Load the model
# ----------------------------------------------------------
MODEL_PATH = "model.h5"

@st.cache_resource(show_spinner=True)
def load_emotion_model():
    model = load_model(MODEL_PATH)
    return model

with st.spinner("Loading model..."):
    model = load_emotion_model()

# Common FER2013 emotion labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------
def load_detector():
    if USE_MTCNN:
        return MTCNN()
    else:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        return cv2.CascadeClassifier(cascade_path)

def detect_faces(image_bgr, detector):
    faces = []
    if USE_MTCNN and hasattr(detector, "detect_faces"):
        detections = detector.detect_faces(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        for d in detections:
            x, y, w, h = d["box"]
            faces.append((x, y, w, h))
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        dets = detector.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in dets:
            faces.append((x, y, w, h))
    return faces

def preprocess_face(face_img):
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (48, 48))
    face_normalized = face_resized.astype("float32") / 255.0
    face_normalized = np.expand_dims(face_normalized, axis=-1)
    face_normalized = np.expand_dims(face_normalized, axis=0)
    return face_normalized

# ----------------------------------------------------------
# Upload or capture image
# ----------------------------------------------------------
option = st.radio("Choose Input Method", ["📤 Upload Image", "📸 Use Camera"])

image_data = None
if option == "📤 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_data = uploaded_file.read()
else:
    camera_file = st.camera_input("Take a photo")
    if camera_file is not None:
        image_data = camera_file.getvalue()

if image_data is not None:
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    detector = load_detector()
    faces = detect_faces(image_bgr, detector)

    if len(faces) == 0:
        st.warning("No faces detected! Try another image or better lighting.")
    else:
        for (x, y, w, h) in faces:
            face = image_bgr[y:y+h, x:x+w]
            input_face = preprocess_face(face)
            preds = model.predict(input_face, verbose=0)
            emotion_idx = np.argmax(preds[0])
            emotion = EMOTIONS[emotion_idx]
            confidence = float(np.max(preds[0]))

            cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image_bgr, f"{emotion} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                 caption="Detected Emotion(s)",
                 use_container_width=True)

        # Show detailed prediction probabilities for the first face
        st.subheader("Emotion Probabilities")
        probs = preds[0]
        for i, e in enumerate(EMOTIONS):
            st.write(f"{e}: {probs[i]:.3f}")
else:
    st.info("Please upload or capture an image to continue.")
