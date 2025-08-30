### Fashion Recommendation System 👕

This project implements a content-based recommendation system that suggests similar fashion items using deep learning. By leveraging the powerful **VGG16** architecture for feature extraction and **cosine similarity** for matching, the system can analyze an outfit image and recommend the most visually similar items from a dataset.

[Demo video](https://drive.google.com/file/d/1KssveIf1Vpn30Zw-WQg5ZUamWPiT7e_f/view?usp=sharing)

-----

### Key Features

  * **Deep Feature Extraction:** Utilizes a pre-trained **VGG16** Convolutional Neural Network (CNN) to extract rich, high-level feature vectors from fashion images.
  * **Similarity-Based Recommendations:** Employs **cosine similarity** to accurately measure the likeness between image features, ensuring relevant recommendations.
  * **Automated Pipeline:** An end-to-end Python pipeline automates the entire workflow, including image preprocessing, feature extraction, and visualization of the recommended outfits.
  * **Scalable Processing:** Efficiently processes and indexes a dataset of 97+ images to provide the top-N most similar recommendations for any given input.

-----

### How It Works

The recommendation process follows a simple yet effective pipeline:

1.  **Image Preprocessing:** All images in the dataset are loaded, resized, and normalized to fit the input requirements of the VGG16 model.
2.  **Feature Extraction:** The VGG16 model, pre-trained on ImageNet, processes each image to generate a unique feature vector (embedding) that captures its visual essence.
3.  **Similarity Calculation:** When a user provides an input image, its feature vector is extracted and compared against the feature vectors of all items in the dataset using cosine similarity.
4.  **Recommendation & Visualization:** The system identifies and returns the top-N items with the highest similarity scores, which are then displayed to the user.

-----

### Tech Stack

  * **Core Language:** **Python**
  * **Deep Learning:** **TensorFlow / Keras** (for VGG16 model)
  * **Numerical & Scientific Computing:** **NumPy**, **SciPy** (for cosine similarity)
  * **Data Visualization:** **Matplotlib**
  * **Image Processing:** **Pillow** / **OpenCV**

-----

### Setup and Usage

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/shinewithsachin/Fashion-Recommendation-System.git
    cd Fashion-Recommendation-System
    ```

2.  **Set Up a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare the Dataset:**

      * Create a folder named `data/images/`.
      * Place all your fashion product images inside this folder.

5.  **Run the Pipeline:**

      * Execute the main script to start the preprocessing and feature extraction process.

    <!-- end list -->

    ```bash
    python main.py --input-image "path/to/your/image.jpg"
    ```

-----

## 🖼 Screenshots

![Input-Output](https://github.com/shinewithsachin/Fashion-Recommendation-System/blob/main/Screenshot%202025-08-25%20225136.png)
![Input-Output](https://github.com/shinewithsachin/Fashion-Recommendation-System/blob/main/Screenshot%202025-08-25%20225200.png).
![Input-Output](https://github.com/shinewithsachin/Fashion-Recommendation-System/blob/main/Screenshot%202025-08-25%20225307.png)
![Input-Output](https://github.com/shinewithsachin/Fashion-Recommendation-System/blob/main/Screenshot%202025-08-25%20225441.png).

-----

### Contributing

Contributions are welcome\! If you have suggestions for improvements, please open an issue or submit a pull request.

-----

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
