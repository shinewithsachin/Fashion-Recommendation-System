import streamlit as st
import os
import numpy as np
from PIL import Image
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from scipy.spatial.distance import cosine

st.set_page_config(page_title="Fashion Recommender", layout="wide")

st.title("👗 Fashion Recommendation System")

st.markdown("""
Welcome! Our collection is focused on **women's fashion**, including dresses, gowns, and traditional wear.
Upload an image of a similar item, and our AI will find the best matches from our collection.
""")

FEATURES_PATH = 'features'
DATASET_PATH = os.path.join('static', 'dataset')
UPLOADS_PATH = os.path.join('static', 'uploads')
os.makedirs(UPLOADS_PATH, exist_ok=True)

st.subheader("Our Collection Includes:")
example_images = [
    'image_01.jpg',
    'image_02.jpg',
    'image_03.jpeg',
    'image_04.jpg'
]

valid_examples = [img for img in example_images if os.path.exists(os.path.join(DATASET_PATH, img))]

if valid_examples:
    cols = st.columns(len(valid_examples))
    for i, col in enumerate(cols):
        with col:
            image_path = os.path.join(DATASET_PATH, valid_examples[i])
            st.image(image_path, use_column_width=True)
else:
    st.warning("Could not find example images. Please check the filenames in `app.py`.")


@st.cache_resource
def load_features():
    """
    Loads the pre-computed image features and filenames from disk.
    """
    try:
        with open(os.path.join(FEATURES_PATH, 'features.pkl'), 'rb') as f:
            feature_list = pickle.load(f)
        with open(os.path.join(FEATURES_PATH, 'image_names.pkl'), 'rb') as f:
            image_names = pickle.load(f)
        return feature_list, image_names
    except FileNotFoundError:
        st.error("Feature files not found. Please run `feature_extractor.py` first.")
        return None, None

feature_list, image_names = load_features()

@st.cache_resource
def load_model():
    """
    Loads the VGG16 model for feature extraction.
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)
    return model

model = load_model()

def save_uploaded_file(uploaded_file):
    """
    Saves the uploaded file to the uploads directory.
    """
    try:
        file_path = os.path.join(UPLOADS_PATH, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def extract_features_from_image(img_path, model):
    """
    Extracts features from a single uploaded image.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img).flatten()
    normalized_features = features / np.linalg.norm(features)
    return normalized_features

def recommend_items(input_features, feature_list, image_names, top_n=5):
    """
    Recommends top_n similar items based on cosine similarity.
    """
    similarities = [1 - cosine(input_features, feature) for feature in feature_list]
    similar_indices = np.argsort(similarities)[-top_n-1:-1][::-1]
    return [image_names[i] for i in similar_indices]

st.subheader("Try It Yourself!")
if feature_list is not None:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        if file_path:
            st.subheader("Your Uploaded Image")
            display_image = Image.open(file_path)
            st.image(display_image, width=250)

            with st.spinner("Finding recommendations..."):
                input_features = extract_features_from_image(file_path, model)
                recommended_images = recommend_items(input_features, feature_list, image_names, top_n=5)

            st.subheader("Here are our top recommendations for you! ✨")
            cols = st.columns(5)
            for i, col in enumerate(cols):
                if i < len(recommended_images):
                    with col:
                        image_path = os.path.join(DATASET_PATH, recommended_images[i])
                        st.image(image_path, caption=f"Recommendation {i+1}", use_column_width=True)