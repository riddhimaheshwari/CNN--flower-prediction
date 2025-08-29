import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained Keras model
@st.cache_resource
def load_my_model():
    # Make sure the .h5 file is in the same directory as your Streamlit script
    model = tf.keras.models.load_model('flower_recognition_model.h5')
    return model

# The class names for mapping predictions
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load the model
model = load_my_model()

st.title('Flower Recognition App')
st.write('Upload a flower image to get a prediction.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    img_array = tf.keras.preprocessing.image.img_to_array(image.resize((150, 150)))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 # Normalize the pixel values
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.write(f"This image most likely belongs to **{predicted_class}** with a {confidence:.2f}% confidence.")
