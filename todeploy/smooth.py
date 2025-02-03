import streamlit as st
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_image(path):
    img = Image.open(path).convert('L')  # Convert to grayscale
    return np.array(img) / 255.0

def create_gaussian_kernel(size, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    x = np.linspace(-(size//2), size//2, size)
    y = x[:, np.newaxis]
    kernel = np.exp(-(x*x + y*y) / (2*sigma*sigma))
    kernel = kernel / kernel.sum()
    return kernel, x

def calculate_snr(original, noisy):
    signal_power = np.mean(original**2)
    noise = noisy - original
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power/noise_power)

def plot_kernel_with_cross_section(kernel, x):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    
    # 2D kernel
    im = ax1.imshow(kernel)
    ax1.set_title('2D Gaussian Kernel')
    plt.colorbar(im, ax=ax1)
    
    # Cross section
    center_row = kernel[kernel.shape[0]//2]
    ax2.plot(x, center_row)
    ax2.set_title('Kernel Cross Section')
    ax2.grid(True)
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Value')
    
    plt.tight_layout()
    return fig

st.title("Neuroimaging Smoothing Demonstration")

# Image selection
st.sidebar.header("Input")
demo_type = st.sidebar.radio("Select input type:", ["Upload Image", "Default Circle"])

if demo_type == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        original = load_image(uploaded_file)
    else:
        st.warning("Please upload an image")
        st.stop()
else:
    # Create circle
    size = 100
    x, y = np.ogrid[-size/2:size/2, -size/2:size/2]
    mask = x*x + y*y <= (size/4)**2
    original = np.zeros((size, size))
    original[mask] = 1

# Parameters
st.sidebar.header("Parameters")
fwhm = st.sidebar.slider("Smoothing FWHM (pixels)", 1, 20, 5)
noise_level = st.sidebar.slider("Noise Level", 0.1, 1.0, 0.5)

# Process images
noisy = original + np.random.normal(0, noise_level, original.shape)
kernel, x = create_gaussian_kernel(20, fwhm)
smoothed = gaussian_filter(noisy, fwhm/2.355)

# Calculate SNR
original_snr = calculate_snr(original, noisy)
smoothed_snr = calculate_snr(original, smoothed)

# Display images
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original Signal")
    st.image(original, clamp=True)

with col2:
    st.subheader(f"Noisy (SNR: {original_snr:.1f}dB)")
    st.image(noisy, clamp=True)

with col3:
    st.subheader(f"Smoothed (SNR: {smoothed_snr:.1f}dB)")
    st.image(smoothed, clamp=True)

# Display kernel with cross section
st.subheader("Smoothing Kernel")
kernel_fig = plot_kernel_with_cross_section(kernel, x)
st.pyplot(kernel_fig)