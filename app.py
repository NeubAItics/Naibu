import os
import cv2
import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_cropper import st_cropper
from PIL import Image
from scipy.sparse import csr_matrix

# Define the list of classes
CLASSES = [
    'A1', 'A2', 'A3', 'A3.5', 'A4', 'B1', 'B2', 'B3', 'B4', 
    'C1', 'C2', 'C3', 'C4', 'D2', 'D3', 'D4'
]

# Function to compute color histogram
def compute_color_histogram(image):
    chans = cv2.split(image)
    hist_values = []
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist_values.extend(hist)
    return hist_values

# Function to compute color moments
def compute_color_moments(image):
    chans = cv2.split(image)
    moments = []
    for chan in chans:
        mean = np.mean(chan)
        std = np.std(chan)
        moments.extend([mean, std])
    return moments

# Preprocess image
def preprocess_image(image):
    if image is None:
        st.error("Failed to load image")
        return None
    hist_values = compute_color_histogram(image)
    color_moments = compute_color_moments(image)
    return hist_values + color_moments

# Load model and predict
def load_model_and_predict(X):
    # Load the model files
    with open('model_files/SVM_model_19-7.pkl', 'rb') as f:
        best_svm_model = pickle.load(f)
    with open('model_files/label_encoder_SVM_19_7.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('model_files/svd_transformer_SVM_19_7.pkl', 'rb') as f:
        svd = pickle.load(f)

    # Transform input
    X_sparse = csr_matrix(X)
    X_reduced = svd.transform(X_sparse)

    # Predict probabilities
    decision_function = best_svm_model.decision_function(X_reduced)
    probabilities = np.exp(decision_function) / np.sum(np.exp(decision_function), axis=1, keepdims=True)
    return probabilities, label_encoder

# Plot predictions
def plot_predictions(image, image_name, top3_classes, top3_probs):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title(image_name)
    ax[0].axis('off')

    bars = ax[1].barh(top3_classes, top3_probs, color='blue')
    ax[1].set_xlabel('Probability')
    ax[1].set_title('Top 3 Predictions')

    for bar, prob in zip(bars, top3_probs):
        ax[1].text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{prob:.2%}', va='center', ha='left')

    st.pyplot(fig)

# Main Streamlit app
def main():
    st.title('Dental Shade Classification')
    st.write("Upload an image of a tooth, crop the area of interest, and classify its shade.")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        st.write(f"Processing image: {uploaded_file.name}")
        image = Image.open(uploaded_file)

        # Display original image
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Cropping section
        st.write("Crop the image below:")
        cropped_img = st_cropper(image, realtime_update=True, box_color='#0D00FF', aspect_ratio=(1, 1))

        # Show cropped image preview
        st.write("Preview of the cropped image:")
        st.image(cropped_img, caption="Cropped Image", use_container_width=True)

        # Predict button
        if st.button("Predict"):
            # Convert cropped image to OpenCV format for preprocessing
            cropped_img_cv = np.array(cropped_img)
            if len(cropped_img_cv.shape) == 3:  # Ensure it's a color image
                cropped_img_cv = cv2.cvtColor(cropped_img_cv, cv2.COLOR_RGB2BGR)

            # Preprocess and predict
            X_new = preprocess_image(cropped_img_cv)
            if X_new is not None:
                X_new = np.array([X_new])
                probabilities, label_encoder = load_model_and_predict(X_new)

                # Extract top 3 predictions
                classes = label_encoder.classes_
                top3_idx = np.argsort(probabilities[0])[::-1][:3]
                top3_classes = classes[top3_idx]
                top3_probs = probabilities[0][top3_idx]

                # Display predictions
                st.write("### Predictions")
                for i in range(3):
                    st.write(f"{top3_classes[i]}: {top3_probs[i]:.2%}")

                # Plot predictions
                plot_predictions(cropped_img_cv, uploaded_file.name, top3_classes, top3_probs)

if __name__ == "__main__":
    main()