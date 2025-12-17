# ================= ENV SAFETY =================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import cv2
import numpy as np
import pickle
import mediapipe as mp
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ================= CONFIG =================
STABILITY_FRAMES = 20
CONFIDENCE_THRESHOLD = 0.85

st.set_page_config(page_title="AI Sign Language Translator", layout="centered")
st.title("🤟 AI Sign Language Translator")

# ================= SAFE LOADERS =================
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("sign_language_model.keras")
    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder


@st.cache_resource
def load_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return hands, mp.solutions.drawing_utils, mp_hands


# ================= LOAD RESOURCES =================
try:
    model, encoder = load_model_and_encoder()
    hands, mp_drawing, mp_hands = load_mediapipe()
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# ================= VIDEO PROCESSOR =================
class SignRecognizer(VideoTransformerBase):
    def __init__(self):
        self.sentence = []
        self.last_pred = None
        self.frame_counter = 0

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        predicted_char = "Waiting..."
        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                )

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

                input_data = np.array([landmarks])
                prediction = model.predict(input_data, verbose=0)

                class_id = np.argmax(prediction)
                confidence = np.max(prediction)

                if confidence > CONFIDENCE_THRESHOLD:
                    predicted_char = encoder.inverse_transform([class_id])[0]

                    if predicted_char == self.last_pred:
                        self.frame_counter += 1
                    else:
                        self.frame_counter = 0
                        self.last_pred = predicted_char

                    if self.frame_counter >= STABILITY_FRAMES:
                        p = predicted_char.lower()
                        if "space" in p:
                            self.sentence.append(" ")
                        elif "delete" in p:
                            if self.sentence:
                                self.sentence.pop()
                        else:
                            self.sentence.append(predicted_char)

                        self.frame_counter = 0

        # ================= UI OVERLAY =================
        h, w, _ = image.shape
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)

        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.circle(image, (w - 30, 30), 15, status_color, -1)

        cv2.putText(image, "DETECTING:", (20, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1)

        cv2.putText(image, predicted_char if hand_detected else "...",
                    (160, 35), cv2.FONT_HERSHEY_DUPLEX, 1,
                    (0, 255, 255), 2)

        sentence_text = "".join(self.sentence)[-25:]
        cv2.putText(image, "SENTENCE:", (20, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1)

        cv2.putText(image, sentence_text,
                    (160, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                    (0, 255, 0), 2)

        if hand_detected and self.last_pred:
            bar_width = int((self.frame_counter / STABILITY_FRAMES) * 100)
            cv2.rectangle(image, (w - 140, 60), (w - 40, 70), (100, 100, 100), -1)
            cv2.rectangle(image, (w - 140, 60),
                          (w - 140 + bar_width, 70),
                          (0, 255, 255), -1)

        return image


# ================= START CAMERA =================
st.markdown("### 🎥 Camera Control")

start = st.button("▶ Start Camera")

if start:
    webrtc_streamer(
        key="sign-language",
        video_transformer_factory=SignRecognizer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
else:
    st.info("Click **Start Camera** to begin detection")