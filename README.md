# Naibu - Dental Shade Classification (Private Repository)

Naibu is a private tool designed for dental professionals to classify dental shades from images using advanced machine learning models. This project is intended for internal use within our organization and should not be shared publicly.

---

## Features

- **Image Upload and Cropping**: Upload dental images and crop the desired region for analysis.
- **Dental Shade Classification**: Predict the dental shade using a pre-trained SVM model.
- **Visualization**: Display top predictions with their probabilities and visual cues.
- **Secure and Simple**: Streamlined for internal deployment, focusing on secure, private usage.

---

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: Support Vector Machine (SVM)
- **Libraries**:
  - `opencv-python-headless`: For image processing
  - `Pillow`: For handling image uploads
  - `streamlit`: For creating the web interface
  - `matplotlib`: For data visualization
  - `scikit-learn`: For machine learning

---

## Prerequisites

Before you begin, ensure you have the following installed:

1. Python 3.9 or later
2. Pip (Python package manager)

---

## Installation

1. Clone the repository (ensure you have access):
   ```bash
   git clone https://github.com/NeubAItics/naibu.git
   cd naibu
   ```

2. Create a virtual environment:
   ```bash
   python -m venv naibu-env
   source naibu-env/bin/activate    # On Windows: naibu-env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open the app in your browser at:
   ```
   http://localhost:8501
   ```

3. Upload an image and crop the desired region using the cropping tool.

4. Click on the **Predict** button to view the results.

---

## File Structure

```
naibu/
├── app.py                   # Main Streamlit application file
├── model_files/             # Contains pre-trained models and transformers
│   ├── SVM_model_19-7.pkl           # Trained SVM model
│   ├── label_encoder_SVM_19_7.pkl   # Label encoder for dental shades
│   └── svd_transformer_SVM_19_7.pkl # SVD transformer
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## Deployment (Internal Use Only)

### Streamlit Sharing

1. Ensure all dependencies in `requirements.txt` are up to date:
   ```bash
   pip freeze > requirements.txt
   ```

2. Deploy the app to a private Streamlit Sharing workspace.

3. Test the app thoroughly in a staging environment before deployment.

---

## Notes

### Security Considerations
1. This repository contains sensitive pre-trained models and internal tools. Do **not** share this repository or its contents externally.
2. Use the internal deployment process to restrict access.

---

## Troubleshooting

### Common Errors and Fixes

#### **1. `libGL.so.1: cannot open shared object file`**
   - Use `opencv-python-headless` instead of `opencv-python`.

#### **2. Dependency Issues**
   - Ensure all dependencies in `requirements.txt` are installed:
     ```bash
     pip install -r requirements.txt
     ```

---

## Contribution Guidelines

This is a private repository for internal use. If you want to propose updates or fixes, please contact the repository owner.

1. Clone the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your message"
   ```
4. Push the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Inform the team about your changes for review.

---

## License

This project is **private** and governed by the organization’s internal policies.

---

## Acknowledgments

Special thanks to the internal team for their efforts in building this project.
