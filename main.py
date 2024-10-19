import streamlit as st
import os
from utils import load_model, predict_image, get_default_device, stats

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
            color: #333;
        }
        .title {
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            color: #3D5B92;
        }
        .description {
            font-size: 1.2em;
            text-align: center;
            margin-bottom: 20px;
            color: #666;
        }
        .predicted {
            font-size: 1.5em;
            font-weight: bold;
            color: #3D5B92;
        }
        .info {
            background-color: #e8f0fe;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("Bird Species Classification", anchor="bird-classification")
    st.write("Upload an image of a bird to classify its species and get detailed information.", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")
        
        device = get_default_device()
        model_path = os.path.join(os.getcwd(), 'model', 'bird-resnet34best.pth')
        model = load_model(model_path, device)
        bird_name, probability, image = predict_image(uploaded_file, model, device, stats)
        
        st.markdown(f"<div class='predicted'>Predicted: {bird_name} with a probability of {round(probability * 100, 2)}%</div>", unsafe_allow_html=True)

        # Link for more information
        st.markdown("[Information about this bird species](https://www.google.com/search?q=bird+species+" + bird_name.replace(" ", "+") + ")", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
