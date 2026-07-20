# pip install opencv-python
# pip install streamlit-drawable-canvas

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model



st.markdown(""" <style>
            .stApp 
            {
                background-color : #d6f0ff;
            } </style>
            """,unsafe_allow_html = True)

st.sidebar.markdown("""
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #163252;   /* Purple */
}
</style>
""", unsafe_allow_html=True)


# Load the pre-trained model
model = load_model('digit_recognition_model.keras')

st.markdown("""<h1 style = 'color : #163252;'>✏️ Handwritten Digit Recognition</h1>""",unsafe_allow_html=True)
st.caption("")
st.caption("")
st.markdown("""<h5 style = 'color : #163252;'>•  Write any no. between 0 to 9 and identify it.</h5>""",unsafe_allow_html=True)
st.sidebar.markdown("""<h1 style = 'color : #d6f0ff;'><u>About Project</u></h1>""",unsafe_allow_html=True)
st.caption("")
st.sidebar.markdown("""<h4 style = 'color : #badaff;'>• This basic project is created for identifying the digits drawn manually by user. </h4>""",unsafe_allow_html=True)
st.sidebar.markdown("""<h4 style = 'color : #badaff;'>• Using the AI/DS concept of ANN (Artificial Neural Network).</h4>""",unsafe_allow_html=True)
st.sidebar.markdown("""<h4 style = 'color : #badaff;'>• Libraries like Tensorflow are used in this model.</h4>""",unsafe_allow_html=True)


canvas_result = st_canvas(
    fill_color = "#00000000",  # Canvas background color -> black
    stroke_width = 10,
    stroke_color ="#FFFFFF",  # Stroke color -> white
    background_color ="#FFFFFFF",
    width = 280,
    height = 280,
    drawing_mode = "freedraw",
    key = "canvas",
)
if st.button("Predict"):
    st.write("Predicting...")

    # Convert the canvas image to a numpy array
    img = canvas_result.image_data.astype(np.uint8)

    #Convert image to greyscale
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize the image to 28x28 pixels
    grey_img = cv2.resize(grey_img, (28, 28))    

    # Normalize the pixel values to be between 0 and 1
    grey_img = grey_img / 255.0

    # Reshape the image to match the input shape of the model
    grey_img = grey_img.reshape(-1,784)

    result = model.predict(grey_img)    # Predict the digit using the pre-trained model

    index = np.argmax(result)   # Get the index of the highest probability digit
    confidence = np.max(result) * 100
    st.markdown(f"""<h4 style = 'color : #163252;'>• Confidence: {confidence:.2f}%</h4>""",unsafe_allow_html=True)

    st.markdown(f"""
        <div style="
        background: linear-gradient(135deg,#2ecc71,#27ae60);
        padding:15px;
        border-radius:12px;
        color:white;
        font-size:18px;
        font-weight:bold;
        box-shadow:0px 4px 12px rgba(0,0,0,0.2);
        ">
        The predicted digit is: {index}
        </div>
        """, unsafe_allow_html=True)
