import streamlit as st

st.title("Traffic Sign Recognition App")

st.header("Upload an image to get started!")

uploaded_file = st.file_uploader("Choose a traffic sign image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("Prediction will appear here...")