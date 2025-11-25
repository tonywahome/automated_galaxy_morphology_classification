# ui/app.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import time

API_URL = "http://localhost:8000"

CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Smooth",
    "Cigar-shaped", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on No Bulge", "Edge-on With Bulge"
]

st.set_page_config(
    page_title="GalaxyAI - Galaxy Classifier",
    page_icon="🌌",
    layout="wide"
)

# === SIDEBAR - MODEL STATUS ===
with st.sidebar:
    st.title("GalaxyAI")
    st.markdown("---")
    
    # Model Uptime
    st.subheader("📊 Model Status")
    try:
        health = requests.get(f"{API_URL}/health").json()
        st.success(f"Status: {health['status'].upper()}")
        st.metric("Uptime", health['uptime_formatted'])
        st.caption(f"Model loaded: {health['model_load_time'][:19]}")
    except:
        st.error("API Unreachable")
    
    st.markdown("---")
    
    # Model Info
    st.subheader("🔬 Model Info")
    try:
        info = requests.get(f"{API_URL}/model/info").json()
        st.write(f"**Version:** {info.get('version', 'N/A')}")
        st.write(f"**Accuracy:** {info.get('accuracy', 'N/A'):.2%}" if isinstance(info.get('accuracy'), float) else "N/A")
        st.write(f"**Classes:** {info.get('classes', 10)}")
    except:
        st.write("Unable to load model info")

# === MAIN TABS ===
tab1, tab2, tab3 = st.tabs([
    "🔮 Predict", "📤 Upload Data", "🔄 Retrain"
])

# === TAB 1: PREDICTION ===
with tab1:
    st.header("Galaxy Classification")
    st.write("Upload a galaxy image to classify its morphological type.")
    
    uploaded_file = st.file_uploader(
        "Choose a galaxy image...", 
        type=['png', 'jpg', 'jpeg'],
        key="predict_upload"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔮 Classify Galaxy", type="primary"):
                with st.spinner("Analyzing galaxy..."):
                    files = {"image": uploaded_file.getvalue()}
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={"image": (uploaded_file.name, uploaded_file.getvalue())}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.prediction = result
    
    with col2:
        if 'prediction' in st.session_state:
            result = st.session_state.prediction
            pred = result['prediction']
            
            st.success(f"**{pred['class_name']}**")
            st.metric("Confidence", f"{pred['confidence']:.1%}")
            st.caption(f"Latency: {result['latency_ms']:.1f}ms")
            
            # Probability chart
            probs = result['probabilities']
            fig = px.bar(
                x=list(probs.values()),
                y=list(probs.keys()),
                orientation='h',
                title="Class Probabilities"
            )
            fig.update_layout(xaxis_title="Probability", yaxis_title="Class")
            st.plotly_chart(fig, use_container_width=True)

# === TAB 2: UPLOAD DATA ===
with tab2:
    st.header("Upload Training Data")
    st.write("Upload new labeled galaxy images to expand the training dataset.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        class_id = st.selectbox(
            "Select Galaxy Class",
            options=list(range(10)),
            format_func=lambda x: f"{x}: {CLASS_NAMES[x]}"
        )
    
    with col2:
        uploaded_files = st.file_uploader(
            "Upload images (bulk upload supported)",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="bulk_upload"
        )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} images for class: **{CLASS_NAMES[class_id]}**")
        
        # Preview
        cols = st.columns(min(5, len(uploaded_files)))
        for i, (col, file) in enumerate(zip(cols, uploaded_files[:5])):
            col.image(file, caption=file.name[:15], width=100)
        
        if st.button("📤 Upload All", type="primary"):
            with st.spinner("Uploading..."):
                files = [("images", (f.name, f.getvalue(), "image/jpeg")) for f in uploaded_files]
                response = requests.post(
                    f"{API_URL}/upload/train-data?class_id={class_id}",
                    files=files
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ Uploaded {result['files_saved']} images!")
                else:
                    st.error("Upload failed")
    
    # Upload Status
    st.markdown("---")
    st.subheader("📊 Current Upload Status")
    try:
        status = requests.get(f"{API_URL}/upload/status").json()
        st.metric("Total Images Uploaded", status['total_uploaded'])
        
        if status['by_class']:
            st.write("**By Class:**")
            for cls, count in status['by_class'].items():
                st.write(f"- {cls}: {count} images")
    except:
        st.write("Unable to fetch upload status")

# === TAB 3: RETRAIN ===
with tab3:
    st.header("Model Retraining")
    st.write("Trigger retraining with newly uploaded data.")
    
    # Current status
    try:
        status = requests.get(f"{API_URL}/retrain/status").json()
        
        if status['is_training']:
            st.warning("🔄 Retraining in progress...")
            progress = st.progress(status['progress'] / 100)
            st.write(f"Status: {status['message']}")
            
            # Stop button when training is in progress
            if st.button("⏹️ Interrupt Training", type="secondary"):
                try:
                    stop_response = requests.post(f"{API_URL}/retrain/stop")
                    if stop_response.status_code == 200:
                        st.warning("⚠️ Training interruption requested...")
                        time.sleep(1)
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to stop training: {e}")
            
            # Auto-refresh
            time.sleep(2)
            st.rerun()
        else:
            if status.get('result'):
                st.success(f"✅ Last retraining complete! Accuracy: {status['result'].get('accuracy', 'N/A'):.2%}")
            elif status.get('message') and 'interrupted' in status['message'].lower():
                st.warning(f"⚠️ Training was interrupted: {status['message']}")
    except:
        pass
    
    # Retraining controls
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Training Epochs", min_value=10, max_value=100, value=30)
    
    with col2:
        st.write("")  # Spacer
        st.write("")
        # Disable start button if training is in progress
        try:
            status = requests.get(f"{API_URL}/retrain/status").json()
            is_training = status.get('is_training', False)
        except:
            is_training = False
            
        if st.button("🚀 Start Retraining", type="primary", disabled=is_training):
            try:
                response = requests.post(
                    f"{API_URL}/retrain/trigger?epochs={epochs}"
                )
                if response.status_code == 200:
                    st.success("Retraining started!")
                    st.rerun()
                else:
                    st.error(response.json().get('detail', 'Error'))
            except Exception as e:
                st.error(f"Failed to start retraining: {e}")
    
    # Instructions
    st.markdown("---")
    st.info("""
    **Retraining Process:**
    1. Upload labeled images in the 'Upload Data' tab
    2. Select number of training epochs
    3. Click 'Start Retraining'
    4. Monitor progress until complete
    5. Use 'Interrupt Training' to stop if needed
    6. Model automatically updates with new weights
    """)