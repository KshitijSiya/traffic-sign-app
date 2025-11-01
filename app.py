import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
# Import the specific preprocessing function for VGG16
from keras.applications.vgg16 import preprocess_input 

# --- 1. REAL MODEL SECTION ---

# Your new list of 50 class names
class_names = [
  "ALL_MOTOR_VEHICLE_PROHIBITED", "AXLE_LOAD_LIMIT", "CATTLE",
  "COMPULSARY_AHEAD_OR_TURN_LEFT", "COMPULSARY_CYCLE_TRACK", "COMPULSARY_KEEP_LEFT",
  "COMPULSARY_MINIMUM_SPEED", "COMPULSARY_SOUND_HORN", "COMPULSARY_TURN_LEFT_AHEAD",
  "COMPULSARY_TURN_RIGHT_AHEAD", "CROSS_ROAD", "CYCLE_CROSSING", "CYCLE_PROHIBITED",
  "FALLING_ROCKS", "HEIGHT_LIMIT", "HORN_PROHIBITED", "HUMP_OR_ROUGH_ROAD",
  "LEFT_HAND_CURVE", "LEFT_REVERSE_BEND", "LEFT_TURN_PROHIBITED", "LENGTH_LIMIT",
  "NARROW_BRIDGE", "NARROW_ROAD_AHEAD", "NO_ENTRY", "NO_PARKING",
  "NO_STOPPING_OR_STANDING", "OVERTAKING_PROHIBITED", "PASS_EITHER_SIDE",
  "PEDESTRIAN_PROHIBITED", "PRIORITY_FOR_ONCOMING_VEHICLES", "QUAY_SIDE_OR_RIVER_BANK",
  "RIGHT_REVERSE_BEND", "ROUNDABOUT", "SCHOOL_AHEAD", "SIDE_ROAD_RIGHT",
  "SLIPPERY_ROAD", "SPEED_LIMIT_30", "SPEED_LIMIT_40", "SPEED_LIMIT_50",
  "SPEED_LIMIT_60", "SPEED_LIMIT_70", "SPEED_LIMIT_80", "STAGGERED_INTERSECTION",
  "STOP", "STRAIGHT_PROHIBITED", "T_INTERSECTION", "U_TURN_PROHIBITED",
  "WIDTH_LIMIT", "Y_INTERSECTION"
]

@st.cache_resource
def load_model():
    # Load the model. Make sure to use your .keras file!
    model = tf.keras.layers.TFSMLayer('saved_model', call_endpoint='serving_default') 
    return model

# Load the model once at the start
model = load_model()

def preprocess_image(image_data):
    """
    Takes an uploaded image, resizes it, and prepares it
    for the VGG16 model.
    """
    # Open the image using PIL
    image = Image.open(image_data)
    
    # Ensure image is in RGB format (VGG16 expects 3 channels)
    image = image.convert('RGB')
    
    # Resize to the model's expected input size
    image = image.resize((224, 224))
    
    # Convert image to a NumPy array
    image_array = np.array(image)
    
    # Add a batch dimension (e.g., shape becomes (1, 224, 224, 3))
    image_array = np.expand_dims(image_array, axis=0)
    
    # Use the official VGG16 preprocessing function
    # This scales pixel values appropriately for the pre-trained model
    preprocessed_image = preprocess_input(image_array)
    
    return preprocessed_image

# --- 2. MAIN APPLICATION UI ---

st.title("Traffic Sign Recognition App")
st.header("Upload an image to get started!")

uploaded_file = st.file_uploader("Choose a traffic sign image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_container_width=True)
    
    # Show a spinner while classifying
    with st.spinner('Classifying...'):
        
        # Preprocess the image
        preprocessed_image = preprocess_image(uploaded_file)
        
        # Get a prediction
        prediction_dict = model(preprocessed_image)

        output_tensor = list(prediction_dict.values())[0]

    # Process the prediction to display the result
    predicted_class_index = np.argmax(output_tensor)
    predicted_class_name = class_names[predicted_class_index]
    confidence_score = np.max(output_tensor) * 100

    # Display the final result
    st.success(f"**Predicted Sign:** {predicted_class_name}")
    st.info(f"**Confidence:** {confidence_score:.2f}%")