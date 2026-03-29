import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model

np.set_printoptions(suppress=True)

@st.cache_resource
def load_tm_model():
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

model, class_names = load_tm_model()

st.set_page_config(page_title="Ghibli or Not Ghibli")

st.title("Ghibli Anime Classifier")
st.write("Upload an image and the app will predict if its ghibli or not.")

uploaded_file = st.file_uploader("Upload an Image: ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)

    # Image processing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = float(prediction[0][index])
    
    # Label
    if len(class_name) > 2 and class_name[1] == " ":
        readable_label = class_name[2:]
    else:
        readable_label = class_name 

    st.subheader("Prediction")
    st.write(f"**Class:** {readable_label}")
    st.write(f"**Confidence:** {confidence_score:.4f}")

    # Show all class probabilities
    st.subheader("All class probabilities")
    probs = prediction[0]
    for i, p in enumerate(probs):
        label_line = class_names[i].strip()
        if len(label_line) > 2 and label_line[1] == " ":
            label = label_line[2:]
        else:
            label = label_line
        st.write(f"{label}: {p:.4f}")