import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2

st.title("🐶 Dog vs Cat Classifier")

model = tf.keras.models.load_model("model.h5")

uploaded_file = st.file_uploader("Upload a cat/dog image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((100, 100))
    img = np.array(img) / 255.0
    img = img.reshape(1, 100, 100, 3)

    pred = model.predict(img)[0][0]
    label = "Dog 🐶" if pred > 0.5 else "Cat 🐱"
    st.subheader(f"Prediction: **{label}**")
