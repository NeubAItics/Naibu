import os
import cv2
import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_cropper import st_cropper
from PIL import Image
from scipy.sparse import csr_matrix
import pandas as pd
import google_sheets_utils as gs
import gspread  
import re

# Define the list of classes
CLASSES = [
    'A1', 'A2', 'A3', 'A3.5', 'A4', 'B1', 'B2', 'B3', 'B4', 
    'C1', 'C2', 'C3', 'C4', 'D2', 'D3', 'D4'
]

def compute_color_histogram(image):
    chans = cv2.split(image)
    hist_values = []
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist_values.extend(hist)
    return hist_values

def compute_color_moments(image):
    chans = cv2.split(image)
    moments = []
    for chan in chans:
        mean = np.mean(chan)
        std = np.std(chan)
        moments.extend([mean, std])
    return moments

def preprocess_image(image):
    if image is None:
        st.error("Failed to load image")
        return None

    hist_values = compute_color_histogram(image)
    color_moments = compute_color_moments(image)
    return hist_values + color_moments

def load_model_and_predict(X):
    with open('model_files/SVM_model_19-7.pkl', 'rb') as f:
        best_svm_model = pickle.load(f)
    with open('model_files/label_encoder_SVM_19_7.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('model_files/svd_transformer_SVM_19_7.pkl', 'rb') as f:
        svd = pickle.load(f)

    X_sparse = csr_matrix(X)
    X_reduced = svd.transform(X_sparse)

    decision_function = best_svm_model.decision_function(X_reduced)
    probabilities = np.exp(decision_function) / np.sum(np.exp(decision_function), axis=1, keepdims=True)
    
    return probabilities, label_encoder

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

def main_pipeline(image):
    X_new = preprocess_image(image)
    
    if X_new is None:
        st.error("Failed to preprocess image")
        return None, None
    
    X_new = np.array([X_new])

    probabilities, label_encoder = load_model_and_predict(X_new)
    
    classes = label_encoder.classes_

    top3_idx = np.argsort(probabilities[0])[::-1][:3]
    top3_classes = classes[top3_idx]
    top3_probs = probabilities[0][top3_idx]

    return top3_classes, top3_probs

def register_user(doctor_name, username, password, email):
    """
    Register a new user and create a new worksheet for the user in Google Sheets with default columns.
    """
    try:
        # Password validation
        if not re.match(r'^[a-zA-Z0-9]{8,}$', password):
            return "Password must be at least 8 characters long and alphanumeric."

        # Load user data from Google Sheets
        users_data = gs.load_users_data()
        
        # Check if the username already exists
        if username in users_data['Username'].values:
            return "Username already exists!"
        
        # Add new user data
        new_user = pd.DataFrame([[doctor_name, username, password, email]], columns=['Doctor Name', 'Username', 'Password', 'Email'])
        users_data = pd.concat([users_data, new_user], ignore_index=True)
        gs.save_users_data(users_data)
        
        # Create a new worksheet for the user
        client = gs.authenticate_google_sheets()
        sheet = client.open_by_key('1lg4nS6uej52cR4QLAn2ZWMoGARpi7fbU4J0GI3avHqY')  # Replace with your actual spreadsheet ID
        new_worksheet = sheet.add_worksheet(title=username, rows="1000", cols="10")
        
        # Define the default column headers
        column_headers = ['Patient Name', 'Tooth Number', 'Predicted Class', 'Actual Class', 'Opinion', 'Comments']
        
        # Update the first row of the new worksheet with column headers
        new_worksheet.update('A1:F1', [column_headers])
        
        return "Registration successful! Please log in."
    
    except gspread.exceptions.SpreadsheetNotFound:
        return "SpreadsheetNotFound exception caught. The specified spreadsheet was not found."
    except gspread.exceptions.WorksheetNotFound:
        return "WorksheetNotFound exception caught. The specified worksheet was not found."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def main():
    st.title('Image Class Prediction with Cropping')

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.predicted_class = None
        st.session_state.uploaded_image = None

    if st.session_state.logged_in:
        username = st.session_state.username
        st.write(f"Welcome, {username}!")

        uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
        if uploaded_file:
            st.session_state.uploaded_image = uploaded_file
            st.write(f"Processing image: {uploaded_file.name}")
            image = Image.open(uploaded_file)
            
            cropped_img = st_cropper(image, realtime_update=True, box_color='#0D00FF', aspect_ratio=(1, 1))
            
            if st.button("Predict"):
                cropped_img = np.array(cropped_img)
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
                top3_classes, top3_probs = main_pipeline(cropped_img)
                
                if top3_classes is not None and top3_probs is not None:
                    top_label = top3_classes[0]
                    top_prob = top3_probs[0]

                    st.write(f"Predicted Class: {top_label} ({top_prob:.2%})")
                    plot_predictions(cropped_img, "Cropped Image", top3_classes, top3_probs)

                    st.session_state.predicted_class = top_label

        if st.session_state.predicted_class:
            with st.form(key='feedback_form'):
                patient_name = st.text_input("Patient Name")
                tooth_number = st.text_input("Tooth Number")
                predicted_class = st.session_state.predicted_class
                actual_class = st.selectbox("Actual Class", CLASSES)
                
                # Add opinion radio buttons
                opinion = st.radio("Do you agree with the prediction?", ["Agree", "Not Agree"])
                
                comments = st.text_area("Comments/Suggestions")

                submit_button = st.form_submit_button("Submit")
                
                if submit_button:
                    if all([patient_name, tooth_number, actual_class, opinion, comments]):
                        st.write("Thank you for your feedback!")
                        try:
                            gs.save_prediction(username, patient_name, tooth_number, predicted_class, actual_class, opinion, comments)
                            st.session_state.predicted_class = None
                        except ValueError as e:
                            st.error(f"Failed to save prediction: {e}")
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {e}")
                    else:
                        st.error("Please fill in all required fields.")
        
        # Add the logout button
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.predicted_class = None
            st.session_state.uploaded_image = None
            st.experimental_rerun()

    else:
        st.header("Login / Register")

        login_option = st.selectbox("Select Option", ["Login", "Register"])
        
        if login_option == "Login":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                users_data = gs.load_users_data()
                user = users_data[(users_data['Username'] == username) & (users_data['Password'] == password)]
                if not user.empty:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        elif login_option == "Register":
            doctor_name = st.text_input("Doctor Name")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            email = st.text_input("Email ID")
            
            if st.button("Register"):
                if all([doctor_name, username, password, email]):
                    result = register_user(doctor_name, username, password, email)
                    if result == "Registration successful! Please log in.":
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error(result)
                else:
                    st.error("Please fill in all required fields.")

if __name__ == '__main__':
    main()
