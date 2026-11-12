import json
from pathlib import Path
from PIL import Image
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Plant Disease AI Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌿 Plant Disease Diagnosis & Health Detection System")
st.markdown("""
This web application uses a **ConvNeXt-Tiny with Squeeze-and-Excitation (SE-Block)** architecture 
trained on over 87,000 leaf images to diagnose **15 distinct plant conditions** across Pepper Bell, Potato, and Tomato.
""")

@st.cache_resource
def get_predictor():
    from predictor import PlantDiseasePredictor
    return PlantDiseasePredictor(bundle_dir=".")

try:
    predictor = get_predictor()
    model_ready = True
except Exception as e:
    st.error(f"Failed to load AI model: {e}")
    model_ready = False

# Sidebar
st.sidebar.header("ℹ️ Model Information")
st.sidebar.markdown("""
- **Backbone:** ConvNeXt-Tiny (Pre-trained)
- **Attention:** Squeeze-and-Excitation (`SE_Block`)
- **Trained Classes:** 15 categories (Pepper, Potato, Tomato)
- **Top Metric:** 72.40% Test Accuracy | 82.75% Recall
- **Input Size:** 128x128 RGB
""")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a plant leaf image (JPG / PNG)...", 
    type=["jpg", "jpeg", "png"]
)

col1, col2 = st.columns([1, 1])

if uploaded_file is not None and model_ready:
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.subheader("📷 Uploaded Leaf Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔬 AI Diagnostic Result")
        with st.spinner("Analyzing leaf pathology..."):
            result = predictor.predict(image, top_k=3)

        plant = result["predicted_plant"]
        diagnosis = result["diagnosis"]
        conf = result["confidence_pct"]
        is_diseased = result["is_diseased"]

        if is_diseased:
            st.error(f"⚠️ **Condition:** {diagnosis}")
            st.markdown(f"**Plant Type:** `{plant}` | **Health Status:** `Diseased`")
        else:
            st.success(f"✅ **Condition:** {diagnosis}")
            st.markdown(f"**Plant Type:** `{plant}` | **Health Status:** `Healthy`")

        st.metric("Model Confidence", f"{conf:.2f}%")

        st.markdown("### Top-3 Predictions:")
        for cand in result["top_k_candidates"]:
            status_icon = "🔴" if cand["is_diseased"] else "🟢"
            st.write(f"{status_icon} **{cand['plant']} - {cand['diagnosis']}**: `{cand['confidence_pct']:.2f}%`")
            st.progress(cand["confidence_pct"] / 100.0)

        with st.expander("🔍 View Raw JSON Diagnostic Metadata"):
            st.json(result)
else:
    st.info("👆 Please upload a photo of a plant leaf above to begin automated diagnosis.")
