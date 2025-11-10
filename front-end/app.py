"""
Streamlit demo application for digit classification.
Usage: streamlit run app.py
"""
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import cv2
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import os

# Page configuration
st.set_page_config(
    page_title="Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-digit {
        font-size: 6rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .confidence {
        font-size: 1.5rem;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """Load and cache the model."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model, None
    except Exception as e:
        return None, str(e)


def preprocess_image(image, target_size=64):
    """Preprocess image for model prediction."""
    # Convert to grayscale if needed
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    # Resize
    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)    
    # Normalize
    image = image.astype('float32') / 255.0
    
    # Add batch and channel dimensions
    image = np.expand_dims(image, axis=(0, -1))
    
    return image


def predict_digit(model, image):
    """Make prediction on preprocessed image."""
    prediction = model.predict(image, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)
    probabilities = prediction[0]
    
    return digit, confidence, probabilities


def plot_probabilities(probabilities):
    """Create probability distribution plot."""
    fig, ax = plt.subplots(figsize=(10, 4))
    digits = list(range(10))
    colors = ['#1f77b4' if p == max(probabilities) else '#aaaaaa' for p in probabilities]
    
    bars = ax.bar(digits, probabilities, color=colors, alpha=0.8)
    ax.set_xlabel('Digit', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Prediction Probabilities', fontsize=14)
    ax.set_xticks(digits)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{prob:.2f}', ha='center', va='bottom', fontsize=10)
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🔢 Digit Classifier</h1>', unsafe_allow_html=True)
    st.markdown("### Draw a digit or upload an image to get predictions")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_options = {
            "Improved CNN (Best)": "./models/improved_cnn_best.h5",
            "Baseline CNN": "./models/baseline_cnn_best.h5",
        }
        
        selected_model_name = st.selectbox(
            "Select Model",
            list(model_options.keys())
        )
        model_path = model_options[selected_model_name]
        
        # Input method
        input_method = st.radio(
            "Input Method",
            ["Draw", "Upload Image"]
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced"):
            img_size = st.slider("Image Size", 28, 128, 64)
            show_preprocessed = st.checkbox("Show Preprocessed Image", value=True)
            show_probabilities = st.checkbox("Show All Probabilities", value=True)
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info(f"**{selected_model_name}**\n\nInput: {img_size}×{img_size} grayscale")
    
    # Load model
    model, error = load_model(model_path)
    
    if error:
        st.error(f"❌ Error loading model: {error}")
        st.info("Make sure you've trained the models first:\n```bash\npython -m src.train --data ./dataset --epochs 15 --img_size 64\n```")
        return
    
    st.success(f"✅ Model loaded: {selected_model_name}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input")
        
        if input_method == "Draw":
            # Drawing canvas
            st.markdown("Draw a digit below:")
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",  # Transparent white fill
                stroke_width=20,
                stroke_color="black",
                background_color="white",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
            
            if canvas_result.image_data is not None:
                # Convert canvas to image
                img = canvas_result.image_data.astype(np.uint8)
                img = canvas_result.image_data
                alpha = img[:, :, 3]  # Use alpha channel as mask
                gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                
                # Use alpha to create clean digit (white on black)
                # Only keep drawn parts (where alpha > 0), set background to 0
                mask = alpha > 30  # Avoid fully transparent
                
                digit = np.zeros_like(gray)
                digit[mask] = 255 - gray[mask]  # black stroke → white digit
            
                # Slight blur for anti-aliasing
                digit = cv2.GaussianBlur(digit, (5, 5), 0)
            
                # Center the digit
                coords = cv2.findNonZero(mask.astype(np.uint8))
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    size = max(w, h, 20)  # min size
                    canvas = np.zeros((size, size), dtype=np.uint8)
                    ox, oy = (size - w) // 2, (size - h) // 2
                    canvas[oy:oy+h, ox:ox+w] = digit[y:y+h, x:x+w]
            
                    # Resize to model size
                    input_image = cv2.resize(canvas, (img_size, img_size), interpolation=cv2.INTER_AREA)
                else:
                    input_image = np.zeros((img_size, img_size), dtype=np.uint8)
        
        else:  # Upload Image
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['png', 'jpg', 'jpeg', 'bmp']
            )
            
            if uploaded_file is not None:
                # Load image
                image = Image.open(uploaded_file).convert('L')
                input_image = np.array(image)
                
                st.image(image, caption="Uploaded Image", use_container_width=True)
            else:
                input_image = None
                st.info("👆 Upload an image to get started")
    
    with col2:
        st.subheader("🎯 Prediction")
        
        if 'input_image' in locals() and input_image is not None:
            # Check if image has any content
            if input_image.max() > 10:  # Some threshold to avoid empty images
                # Preprocess
                processed_image = preprocess_image(input_image, target_size=img_size)
                
                # Show preprocessed image
                if show_preprocessed:
                    st.markdown("**Preprocessed Image:**")
                    st.image(
                        processed_image[0, :, :, 0],
                        caption=f"{img_size}×{img_size} Grayscale",
                        width=200,
                        clamp=True
                    )
                
                # Predict
                with st.spinner("Predicting..."):
                    digit, confidence, probabilities = predict_digit(model, processed_image)
                
                # Display prediction
                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <div class="prediction-digit">{digit}</div>
                        <div class="confidence">Confidence: {confidence:.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Show probability distribution
                if show_probabilities:
                    st.markdown("**Probability Distribution:**")
                    fig = plot_probabilities(probabilities)
                    st.pyplot(fig)
                
                # Additional info
                with st.expander("📈 Detailed Results"):
                    st.markdown("**Top 3 Predictions:**")
                    top_3_idx = np.argsort(probabilities)[-3:][::-1]
                    for idx in top_3_idx:
                        st.write(f"- Digit **{idx}**: {probabilities[idx]:.2%}")
                    
                    st.markdown("**All Probabilities:**")
                    prob_dict = {f"Digit {i}": f"{p:.4f}" for i, p in enumerate(probabilities)}
                    st.json(prob_dict)
            
            else:
                st.info("✏️ Draw something or upload an image to see predictions")
        else:
            st.info("👈 Draw a digit or upload an image to get started")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with Streamlit 🎈 | TensorFlow 🔥</p>
            <p>Model trained on digit classification dataset</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()