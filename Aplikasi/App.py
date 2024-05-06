import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def load_model():
    """
    Load pre-trained deep learning model for carrot freshness classification.
    
    Returns:
        model (tf.keras.Model): Loaded pre-trained deep learning model.
    """
    model = tf.keras.models.load_model('best_model.h5')
    return model

def preprocessing_image(img):
    """
    Preprocesses the input image for model prediction.
    
    Args:
        img: Input image to be preprocessed.
        
    Returns:
        images (numpy.ndarray): Preprocessed image ready for prediction.
    """
    img = img.resize((256, 256))
    images = image.img_to_array(img)
    images /= 255
    images = np.expand_dims(images, axis=0)
    return images

#@st.cache(suppress_st_warning=True)
@st.cache_data
def get_prediction(processed_images):
    """
    Get prediction from the loaded model.
    
    Args:
        processed_images (numpy.ndarray): Preprocessed image for prediction.
        
    Returns:
        freshness (str): Predicted freshness class ('Fresh' or 'Rotten').
        probability (float): Probability of the predicted class.
    """
    classes = model.predict(processed_images, batch_size=16)
    output_class = np.argmax(classes)
    probability = classes[0][output_class]
    classname = ['Fresh', 'Rotten']
    freshness = classname[output_class]
    return freshness, probability

def display_result(freshness, probability):
    """
    Display prediction result.
    
    Args:
        freshness (str): Predicted freshness class ('Fresh' or 'Rotten').
        probability (float): Probability of the predicted class.
    """
    st.image(img, caption='Gambar yang dipilih', use_column_width=False, width=256)
    st.write("Hasil Klasifikasi:")
    st.write(f"Freshness: {freshness}")
    st.write(f"Probability: {probability:.2%}")

# Load pre-trained model
model = load_model()

st.title("Aplikasi Klasifikasi Kesegaran Wortel")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Preprocess the image
    img = image.load_img(uploaded_file)
    processed_images = preprocessing_image(img)

    # Predict Image
    freshness, probability = get_prediction(processed_images)

    # Display prediction result
    display_result(freshness, probability)

st.write("Develop by Putra Al Farizi")
