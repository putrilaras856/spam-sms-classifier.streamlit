import streamlit as st
from joblib import load

import nltk
import os

# Set the NLTK to use the nltk_data folder in the current directory
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

# Set the page config to change the tab title and favicon
st.set_page_config(page_title="Spam SMS Classifier", page_icon="📱", layout="wide")

# Load the saved TfidfVectorizer and model from the .joblib files
vectorizer = load('tfidf_vectorizer.joblib')
model = load('spam_class.joblib')

# Define the label dictionary
label_dict = {0: 'Normal', 1: 'Fraud/Penipuan', 2: 'Promo'}

# Define the label colors for displaying prediction results
label_colors = {
    0: 'green',  # Normal
    1: 'red',    # Fraud/Penipuan
    2: 'blue'    # Promo
}

# Streamlit webpage
st.title('SMS Spam Prediction Tool')
st.write('This tool predicts whether a given SMS message is normal, fraudulent, or promotional.')

# Text box for user input
input_sms = st.text_area("Enter the SMS text you want to analyze:", "")

# Process the user input and make a prediction
def classify_message(model, vectorizer, message):
    processed_message = vectorizer.transform([message])
    prediction = model.predict(processed_message)
    proba = model.predict_proba(processed_message)
    return prediction, proba

# Predict button logic
if st.button('Predict'):
    if input_sms:
        prediction, proba = classify_message(model, vectorizer, input_sms)
        result_label = label_dict[prediction[0]]
        result_color = label_colors[prediction[0]]
        
        # Custom HTML for displaying the classification result
        st.markdown(f"""
            <div style='color: white; font-size: 30px;'><strong>The SMS message is classified as:</strong></div>
            <div style='color: {result_color}; font-size: 36px;'><strong>{result_label}</strong></div>
            <br>
            """, unsafe_allow_html=True)

        
        # Display prediction probabilities as percentages
        st.subheader('Prediction Probabilities:')
        for index, label in enumerate(label_dict.values()):
            st.write(f"{label}: {proba[0][index]*100:.1f}%")
    else:
        st.error("Please enter a SMS text to classify.")

# Footer with contact information or additional details
st.markdown("""
    <hr>
    SMS Spam Prediction Tool by STKI Kelompok 1.
""", unsafe_allow_html=True)
