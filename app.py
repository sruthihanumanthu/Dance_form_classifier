import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import time
from io import BytesIO
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configuration
st.set_page_config(
    page_title="Dance Form Classifier",
    layout="wide",
    page_icon="💃",
    initial_sidebar_state="expanded"
)

# Load model and class names with caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("dance_model.h5")
        # Suppress model compilation warning
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        with open("labels.txt") as f:
            class_names = f.read().splitlines()
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model, class_names = load_model()

# Enhanced prediction function
def predict_dance(image, top_n=3):
    img = np.array(image)
    
    # Preprocessing
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    predictions = model.predict(img, verbose=0)[0]
    
    # Get top predictions
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    top_classes = [class_names[i] for i in top_indices]
    top_confidences = [predictions[i] for i in top_indices]
    
    return top_classes, top_confidences

# Updated Grad-CAM visualization with automatic layer detection
def generate_heatmap(image, model):
    img_array = np.array(image)
    
    # Preprocessing
    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    img_array = cv2.resize(img_array, (128, 128))
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Find the last convolutional layer automatically
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break
    
    if last_conv_layer is None:
        raise ValueError("No convolutional layer found in the model")
    
    # Create gradient model
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer).output, model.output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]
    
    # Generate heatmap
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    
    # Post-process heatmap
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap, last_conv_layer

# UI Components
def main():
    # Sidebar
    with st.sidebar:
        st.title("About")
        st.markdown("""
        This advanced classifier identifies classical dance forms from images.
        **Features:**
        - Multi-class prediction
        - Confidence visualization
        - Heatmap generation
        - Performance metrics
        - Batch processing
        """)
        
        st.divider()
        st.markdown("**Settings**")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
        top_n = st.slider("Show Top Predictions", 1, 5, 3)

        st.divider()
        st.markdown("**Model Info**")
        st.text(f"Input shape: {model.input_shape[1:3]}")
        st.text(f"Classes: {len(class_names)}")
        st.text(f"Last Conv Layer: {[layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]}")
        
        if st.button("Clear Cache"):
            st.cache_resource.clear()
            st.rerun()

    # Main content
    st.title("💃 Dance Form Classifier")
    st.markdown("Upload images of classical dance forms for AI-powered classification.")
    
    tab1, tab2 = st.tabs(["Single Image", "Batch Processing"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image...", 
                type=["jpg", "jpeg", "png"],
                key="single_upload"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='Uploaded Image', use_container_width=True)
                
                if st.button("Analyze", key="analyze_single"):
                    with st.spinner("Processing..."):
                        start_time = time.time()
                        top_classes, top_confidences = predict_dance(image, top_n)
                        processing_time = time.time() - start_time
                        
                        with col2:
                            st.subheader("Analysis Results")
                            
                            # Confidence bar chart
                            fig, ax = plt.subplots()
                            y_pos = np.arange(len(top_classes))
                            ax.barh(y_pos, top_confidences, color='skyblue')
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(top_classes)
                            ax.invert_yaxis()
                            ax.set_xlabel('Confidence')
                            ax.set_title('Top Predictions')
                            st.pyplot(fig)
                            
                            # Primary prediction
                            primary_class = top_classes[0]
                            primary_confidence = top_confidences[0]
                            
                            if primary_confidence < confidence_threshold:
                                st.warning(f"⚠️ Low confidence ({primary_confidence*100:.1f}%) - This might not be a classical dance form.")
                            else:
                                st.success(f"🎯 Primary Prediction: **{primary_class}**")
                               
                            
                            st.info(f"⏱️ Processing time: {processing_time:.2f}s")
                            st.markdown(f"<h1 style='color: #FF4B4B;'>{primary_class.capitalize()}</h1>", unsafe_allow_html=True)
                            # Result section
                            with st.container():
                                st.markdown("<div class='result-section'>", unsafe_allow_html=True)
                                
                                col_res1, col_res2 = st.columns([1, 2])
                                
                                with col_res1:
                                    st.metric("Confidence", f"{primary_confidence*100:.2f}%")
                                    
                                with col_res2:
                                    # Display dance form information
                                    dance_info = {
                                        'bharatanatyam': {
                                            'origin': 'Tamil Nadu',
                                            'features': 'Fixed upper torso, bent legs, intricate footwork, expressive hand gestures',
                                            'costume': 'Colorful saree with pleated cloth in front'
                                        },
                                        'kathak': {
                                            'origin': 'North India',
                                            'features': 'Fast footwork, spins, storytelling through gestures and facial expressions',
                                            'costume': 'Anarkali suit with lehenga for women, kurta-churidar for men'
                                        },
                                        'odissi': {
                                            'origin': 'Odisha',
                                            'features': 'Tribhangi posture (three bends), fluid movements, sculpturesque poses',
                                            'costume': 'Bright colored saree with traditional Odisha printing'
                                        },
                                        'kathakali': {
                                            'origin': 'Kerala',
                                            'features': 'Elaborate makeup, dramatic facial expressions, vibrant costumes',
                                            'costume': 'Large skirts, elaborate headdresses, and colorful makeup'
                                        },
                                        'sattriya': {
                                            'origin': 'Assam',
                                            'features': 'Graceful movements, distinctive costumes, devotional themes',
                                            'costume': 'White or cream dhoti and chadar with red borders'
                                        },
                                        'manipuri': {
                                            'origin': 'Manipur',
                                            'features': 'Smooth, gentle movements, rounded poses, no heavy stepping',
                                            'costume': 'Potloi costume for women with cylindrical skirt'
                                        },
                                        'kuchipudi': {
                                            'origin': 'Andhra Pradesh',
                                            'features': 'Fast rhythmic footwork, sculpturesque body movements',
                                            'costume': 'Bright colored saree with pleats in front'
                                        },
                                        'mohiniyattam': {
                                            'origin': 'Kerala',
                                            'features': 'Graceful, swaying movements, feminine dance theme',
                                            'costume': 'White or off-white saree with golden border'
                                        }
                                    }
                                    
                                    # Convert prediction to lowercase for case-insensitive matching
                                    predicted_class_lower = primary_class.lower()
                                    
                                    if predicted_class_lower in dance_info:
                                        info = dance_info[predicted_class_lower]
                                        st.markdown("<div class='dance-info'>", unsafe_allow_html=True)
                                        st.markdown(f"**Origin:** {info['origin']}")
                                        st.markdown(f"**Key Features:** {info['features']}")
                                        st.markdown(f"**Costume:** {info['costume']}")
                                        st.markdown("</div>", unsafe_allow_html=True)
                                    else:
                                        st.warning(f"No additional information available for {primary_class}")
                                
                                st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Batch Processing")
        batch_files = st.file_uploader(
            "Upload multiple images", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_upload"
        )
        
        if batch_files and st.button("Process Batch"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(batch_files):
                try:
                    status_text.text(f"Processing {i+1}/{len(batch_files)}: {uploaded_file.name}")
                    progress_bar.progress((i + 1) / len(batch_files))
                    
                    image = Image.open(uploaded_file).convert('RGB')
                    top_classes, top_confidences = predict_dance(image, 1)
                    results.append({
                        "filename": uploaded_file.name,
                        "prediction": top_classes[0],
                        "confidence": top_confidences[0],
                        "status": "✅" if top_confidences[0] >= confidence_threshold else "⚠️"
                    })
                except Exception as e:
                    results.append({
                        "filename": uploaded_file.name,
                        "prediction": "Error",
                        "confidence": 0,
                        "status": "❌"
                    })
            
            # Display results
            st.subheader("Batch Results")
            st.dataframe(
                results,
                column_config={
                    "filename": "Filename",
                    "prediction": "Prediction",
                    "confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                    "status": "Status"
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Export results
            csv = "\n".join([f"{r['filename']},{r['prediction']},{r['confidence']}" for r in results])
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="dance_classification_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
