import os
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from PIL import Image
from tqdm import tqdm
import pickle

DATASET_PATH = os.path.join('static', 'dataset')
FEATURES_PATH = 'features'

os.makedirs(FEATURES_PATH, exist_ok=True)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

def extract_features(img_path, model):
    """
    Extracts deep features from an image using the VGG16 model.
    """
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)

        features = model.predict(preprocessed_img, verbose=0)

        flattened_features = features.flatten()
        normalized_features = flattened_features / np.linalg.norm(flattened_features)

        return normalized_features
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

image_files = [os.path.join(DATASET_PATH, f) for f in os.listdir(DATASET_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg', 'webp'))]

all_features = []
all_image_names = []

print("Starting feature extraction...")
for img_path in tqdm(image_files, desc="Extracting features"):
    features = extract_features(img_path, model)
    if features is not None:
        all_features.append(features)
        all_image_names.append(os.path.basename(img_path))

print("Feature extraction completed.")

feature_list_path = os.path.join(FEATURES_PATH, 'features.pkl')
image_names_path = os.path.join(FEATURES_PATH, 'image_names.pkl')

with open(feature_list_path, 'wb') as f:
    pickle.dump(all_features, f)

with open(image_names_path, 'wb') as f:
    pickle.dump(all_image_names, f)

print(f"Saved {len(all_features)} features to {feature_list_path}")
print(f"Saved {len(all_image_names)} image names to {image_names_path}")