import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

# Load model once
@st.cache_data
def load_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    return model

preprocess_func = ResNet50_Weights.IMAGENET1K_V2.transforms()
categories = np.array(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])

def make_prediction(model, processed_img):
    probs = model(processed_img.unsqueeze(0))
    probs = probs.softmax(1)
    probs = probs[0].detach().numpy()

    probs, ind = probs[probs.argsort()[-5:][::-1]], probs.argsort()[-5:][::-1]
    return probs, ind

def interpret_prediction(model, processed_img, target):
    int_algo = IntegratedGradients(model)
    features_imp = int_algo.attribute(processed_img.unsqueeze(0), target=int(target))
    features_imp = features_imp[0].numpy()
    features_imp = features_imp.transpose(1, 2, 0)
    return features_imp

def main():
    st.title("VisualAI: Image Classification App 🤖 🖼️ ")
    upload = st.file_uploader(label="Upload image", type=["jpg", "png", "jpeg"])

    if upload:
        img = Image.open(upload)
        model = load_model()
        processed_img = preprocess_func(img)
        probs, ind = make_prediction(model, processed_img)
        features_imp = interpret_prediction(model, processed_img, ind[0])

        # Displaying top 5 probabilities
        fig1, ax1 = plt.subplots(figsize=(6, 2))
        ax1.barh(y=categories[ind][::-1], width=probs[::-1], color=['red'] * 4 + ['limegreen'])
        ax1.set_title("Top 5 Probabilities", fontsize=12)
        st.pyplot(fig1)

        # Displaying images and plots below
        col1, col2 = st.columns(2)

        # Displaying uploaded image on the left
        with col1:
            st.image(img, caption='Uploaded Image', use_column_width=True)

        # Displaying interpretation picture on the right
        with col2:
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            viz.visualize_image_attr(features_imp, show_colorbar=True, fig_size=(4, 4))
            st.pyplot(fig2)
