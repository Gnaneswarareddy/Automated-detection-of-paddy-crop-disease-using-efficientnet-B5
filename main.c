import streamlit as st
import tensorflow as tf
from keras_preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the saved model
# @st.cache_resource(allow_output_mutation=True)
model = load_model('rice_leaf_disease_modelefficentnet.h5')

# Class labels
class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy', 'Leaf Blast', 'Leaf Scald', 'Narrow Brown Spot']

# Image preprocessing function
def prepare_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescaling
    return img_array

# Streamlit UI
st.title("Automated Detection of Paddy Crop Disease")

# Upload image
uploaded_file = st.file_uploader("Upload an image of a rice leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img_array = prepare_image(uploaded_file.name)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_names[predicted_class]
    predicted_confidence = prediction[0][predicted_class_index] * 100

    # Show the result
    st.info(f"#### Prediction: **{predicted_label}**")
    st.info(f"##### **Confidence of Predicted Class:** {predicted_confidence:.2f}%")
    
    # Show the confidence levels
    confidence_levels = {class_names[i]: f"{round(prediction[0][i] * 100, 2)}%" for i in range(len(class_names))}
    st.write("Confidence Levels:")
    st.json(confidence_levels)
    
