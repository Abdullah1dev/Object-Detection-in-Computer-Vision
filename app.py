import streamlit as st
import cv2
import numpy as np

from src.detector import (
    load_model,
    detect_objects,
    draw_detections
)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def get_model():
    return load_model()


net = get_model()


# ============================================================
# Header
# ============================================================

st.title("🔍 AI Object Detection")
st.markdown(
    """
    Upload an image and let a pretrained **MobileNet-SSD**
    model detect objects inside it.
    """
)

st.divider()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("⚙️ About the Model")

    st.markdown(
        """
        ### MobileNet-SSD

        This application uses a pretrained
        **MobileNet-SSD** object detection model.

        **Model capabilities:**

        - Object detection
        - Bounding boxes
        - Confidence scores
        - Multiple object detection

        **Frameworks:**

        - Python
        - OpenCV
        - OpenCV DNN
        - Streamlit
        """
    )

    st.divider()

    st.info(
        "Upload an image from the main panel to begin detection."
    )


# ============================================================
# Image Upload
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# Process Image
# ============================================================

if uploaded_file is not None:

    # Read uploaded file
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    # Convert bytes into OpenCV image
    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error(
            "❌ Could not read the uploaded image."
        )

    else:

        # Convert BGR → RGB for Streamlit
        original_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ====================================================
        # Run Detection
        # ====================================================

        with st.spinner("🔎 Detecting objects..."):

            results = detect_objects(
                image,
                net
            )

            output_image = draw_detections(
                image,
                results
            )

            output_image = cv2.cvtColor(
                output_image,
                cv2.COLOR_BGR2RGB
            )

        # ====================================================
        # Display Images
        # ====================================================

        st.subheader("🖼️ Detection Result")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Original Image")

            st.image(
                original_image,
                use_container_width=True
            )

        with col2:

            st.markdown("### Detected Objects")

            st.image(
                output_image,
                use_container_width=True
            )

        st.divider()

        # ====================================================
        # Detection Summary
        # ====================================================

        st.subheader("📊 Detection Summary")

        if len(results) == 0:

            st.warning(
                "No objects were detected in this image."
            )

        else:

            st.success(
                f"✅ {len(results)} object(s) detected."
            )

            # =================================================
            # Display Individual Results
            # =================================================

            for index, result in enumerate(results, start=1):

                label = result["label"]
                confidence = result["confidence"]
                box = result["box"]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Object",
                        label.title()
                    )

                with col2:
                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                with col3:
                    st.write("**Bounding Box**")
                    st.code(str(box))

                if index < len(results):
                    st.divider()


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Built with ❤️ using Python, OpenCV, MobileNet-SSD and Streamlit"
)