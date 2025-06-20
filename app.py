import streamlit as st
import pickle

# Load model and tools
model = pickle.load(open('job_category_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

st.title("🧠 Job Category Predictor (India)")

job_title = st.text_input("Enter Job Title")
skills = st.text_input("Enter Skills (comma separated)")
experience = st.text_input("Enter Experience (e.g., 2-4 years)")
location = st.text_input("Enter Location (e.g., Pune)")

if st.button("Predict Category"):
    input_text = job_title + ' ' + skills + ' ' + experience + ' ' + location
    X_input = vectorizer.transform([input_text])
    prediction = model.predict(X_input)
    result = label_encoder.inverse_transform(prediction)
    st.success(f"Predicted Job Category: {result[0]}")


# Run this in powershell:
# & "C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe" -m pip install streamlit scikit-learn
# & "C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run app.py

