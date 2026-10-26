import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "animal_detection_model.keras"
CLASSES_PATH = APP_DIR / "animal_classes.json"

st.set_page_config(
    page_title="AI Animal Detector",
    page_icon="🐾",
    layout="centered",
)

st.title("🐾 AI Animal Detection")
st.write(
    "Upload an animal image and the AI model will predict whether it is a "
    "cat, dog, elephant, lion, or tiger."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def load_classes():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error(
        "Model files are missing. Please add both "
        "`animal_detection_model.keras` and `animal_classes.json` "
        "to the same folder as app.py, then redeploy."
    )
    st.stop()

try:
    model = load_model()
    class_names = load_classes()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an animal image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("🔍 Predict Animal", type="primary"):
            with st.spinner("Analyzing image..."):
                resized = image.resize((224, 224))
                image_array = np.asarray(resized, dtype=np.float32)
                image_array = np.expand_dims(image_array, axis=0)

                predictions = model.predict(image_array, verbose=0)[0]
                best_index = int(np.argmax(predictions))
                predicted_class = class_names[best_index]
                confidence = float(predictions[best_index]) * 100

            st.success(f"Predicted animal: **{predicted_class}**")
            st.metric("Model confidence", f"{confidence:.2f}%")
            st.caption(
                "Educational demo only. The model was trained with a very small "
                "dataset, so predictions may be incorrect, especially for new images."
            )

            with st.expander("Show scores for all classes"):
                for i in np.argsort(predictions)[::-1]:
                    st.write(f"{class_names[int(i)]}: {predictions[int(i)] * 100:.2f}%")
                    st.progress(float(predictions[int(i)]))
    except Exception as e:
        st.error(f"Could not process this image: {e}")

st.divider()
st.caption("AI Animal Detection Project | Built with TensorFlow and Streamlit")
