import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Model
model = tf.keras.models.load_model(
    "model/Plant_Disease_Model.keras"
)

class_names = ["Infected", "Healthy"]

IMG_SIZE = 180

# Page Title
st.title("🌿 Plant Disease Prediction System")

st.write(
    "Upload a plant leaf image to determine whether the plant is healthy or infected."
)

# Upload Image
uploaded_file = st.file_uploader(
    "Upload Plant Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display Image
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocessing
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image)

    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    st.divider()

    # Display Result
    if predicted_class == "Healthy":
        st.success(
            f"Prediction: {predicted_class}"
        )
    else:
        st.error(
            f"Prediction: {predicted_class}"
        )

    st.write(
        f"**Confidence Score:** {confidence:.2f}%"
    )

    st.progress(float(confidence / 100))
