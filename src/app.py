import streamlit as st
import numpy as np
from PIL import Image
import io
import os

from huggingface_hub import hf_hub_download


# Hybrid TFLite Import (Linux = tflite_runtime / Windows = tensorflow.lite)

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
    TFLITE_AVAILABLE = False


def load_interpreter(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


# Streamlit settings
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Dark theme override
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
</style>
""", unsafe_allow_html=True)


# Load model
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="jamirc/cat_dog_classifier",
        filename="model.tflite",
        local_dir=".",
        local_dir_use_symlinks=False
    )
    return load_interpreter(model_path)


interpreter = load_model()

NAMES = ['Cat', 'Dog']


# Image loading
def load_image_from_bytes(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize((200, 200))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array.astype(np.float32), axis=0)


# Prediction function

def predict_image(img_array):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, img_array)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_index)
    idx = int(np.argmax(pred))
    prob = float(pred[0][idx])

    return NAMES[idx], prob



# streamlit UI

st.title("🐾 Cat vs Dog Classifier")
st.write("Upload an image 🐱 or 🐶.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img_bytes = uploaded_file.read()
    img_array = load_image_from_bytes(img_bytes)

    label, prob = predict_image(img_array)

    st.subheader("Prediction")
    st.write(f"**Label:** {label}")
    st.write(f"**Confidence:** {prob:.4f}")

    st.write("### Probability")
    st.progress(prob)
