import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from PIL import Image
import io

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Image Processing Playground",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Image Processing Playground")
st.write("Upload an image and explore different enhancement methods interactively.")


# ----------------------------
# Helper Functions
# ----------------------------
def show_hist(image, is_gray, title="Histogram"):
    """Return a matplotlib histogram figure for grayscale or color image."""
    fig, ax = plt.subplots(figsize=(4, 3))
    if is_gray:
        ax.hist(image.ravel(), bins=256, range=(0, 256), color='black')
    else:
        colors = ('r', 'g', 'b')
        for i, col in enumerate(colors):
            ax.hist(image[:, :, i].ravel(), bins=256, range=(0, 256), color=col, alpha=0.6)
    ax.set_title(title)
    ax.set_xlim([0, 256])
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def apply_transformation(image, method, params, is_gray):
    """Apply selected image processing method."""
    if method == "Linear Negative":
        return 255 - image

    elif method == "Contrast Stretching":
        r_min, r_max = params["r_min"], params["r_max"]
        return np.clip((image - r_min) * (255.0 / (r_max - r_min)), 0, 255).astype(np.uint8)

    elif method == "Piecewise Linear":
        r1, s1, r2, s2 = params["r1"], params["s1"], params["r2"], params["s2"]
        lut = np.interp(np.arange(256),
                        [0, r1, r2, 255],
                        [0, s1, s2, 255]).astype("uint8")
        return cv2.LUT(image, lut)

    elif method == "Log Transformation":
        c = params["c"]
        result = c * np.log1p(image.astype(np.float32))
        return np.uint8(255 * result / np.max(result))

    elif method == "Gamma Transformation":
        gamma = params["gamma"]
        result = np.power(image / 255.0, gamma)
        return np.uint8(result * 255)

    elif method == "Histogram Equalization":
        if is_gray:
            return cv2.equalizeHist(image)
        else:
            img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    elif method == "Adaptive Histogram Equalization":
        clip = params["clip_limit"]
        if is_gray:
            eq = exposure.equalize_adapthist(image / 255.0, clip_limit=clip)
            return np.uint8(eq * 255)
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = np.uint8(exposure.equalize_adapthist(l / 255.0, clip_limit=clip) * 255)
            return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    elif method == "CLAHE":
        clahe = cv2.createCLAHE(clipLimit=params["clip_limit"], tileGridSize=(params["tile_size"], params["tile_size"]))
        if is_gray:
            return clahe.apply(image)
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    return image


def convert_to_bytes(img_array, is_gray):
    """Convert numpy image to bytes for download."""
    img_pil = Image.fromarray(img_array)
    buf = io.BytesIO()
    if is_gray:
        img_pil.convert("L").save(buf, format="PNG")
    else:
        img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    is_gray = st.sidebar.radio("Image Type", ["Color", "Grayscale"]) == "Grayscale"
    if is_gray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    method = st.sidebar.selectbox("Processing Method", [
        "Linear Negative",
        "Contrast Stretching",
        "Piecewise Linear",
        "Log Transformation",
        "Gamma Transformation",
        "Histogram Equalization",
        "Adaptive Histogram Equalization",
        "CLAHE"
    ])

    # Parameters (dynamic)
    params = {}
    if method == "Contrast Stretching":
        params["r_min"] = st.sidebar.slider("r_min", 0, 255, 50)
        params["r_max"] = st.sidebar.slider("r_max", 0, 255, 200)
    elif method == "Piecewise Linear":
        params["r1"] = st.sidebar.slider("r1", 0, 255, 70)
        params["s1"] = st.sidebar.slider("s1", 0, 255, 0)
        params["r2"] = st.sidebar.slider("r2", 0, 255, 140)
        params["s2"] = st.sidebar.slider("s2", 0, 255, 255)
    elif method == "Log Transformation":
        params["c"] = st.sidebar.slider("c (scale)", 1, 50, 20)
    elif method == "Gamma Transformation":
        params["gamma"] = st.sidebar.slider("Gamma", 0.1, 5.0, 1.0, 0.1)
    elif method == "Adaptive Histogram Equalization":
        params["clip_limit"] = st.sidebar.slider("Clip Limit", 0.01, 0.1, 0.03, 0.01)
    elif method == "CLAHE":
        params["clip_limit"] = st.sidebar.slider("Clip Limit", 1.0, 10.0, 2.0)
        params["tile_size"] = st.sidebar.slider("Tile Size", 2, 16, 8)

    # ----------------------------
    # Processing
    # ----------------------------
    result = apply_transformation(image, method, params, is_gray)

    # ----------------------------
    # Layout: Original vs Processed
    # ----------------------------
    st.subheader(f"🔍 Comparing: **Original** vs **{method}**")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, channels="GRAY" if is_gray else "RGB", caption="Original Image", use_container_width=True)
        st.pyplot(show_hist(image, is_gray, "Original Histogram"))
    with col2:
        st.image(result, channels="GRAY" if is_gray else "RGB", caption=f"Processed Image ({method})", use_container_width=True)
        st.pyplot(show_hist(result, is_gray, "Processed Histogram"))

    # ----------------------------
    # Download Button
    # ----------------------------
    st.download_button(
        label="💾 Download Processed Image",
        data=convert_to_bytes(result, is_gray),
        file_name=f"processed_{method.replace(' ', '_').lower()}.png",
        mime="image/png"
    )

else:
    st.info("👆 Upload an image from the sidebar to get started.")
