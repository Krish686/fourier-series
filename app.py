import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set up the page layout
st.set_page_config(page_title="Fourier Series Experiment", layout="wide")

st.title("Fourier Series Simulation (Generalized)")
st.write("Visualizing Fourier Series with adjustable L, N, and dx.")

# Function to perform the numerical calculations
# Uses vectorization (matrix math) for high performance
@st.cache_data(show_spinner=False)
def calculate_fourier(wave_type, L, dx, N):
    # 1. Setup Domain
    # Matches MATLAB logic 'x = dx:dx:L'
    # The small epsilon (+ dx/100000) ensures the endpoint L is included
    x = np.arange(dx, L + dx/100000, dx)
    
    # 2. Setup Waveform
    # Initialize with zeros
    f = np.zeros_like(x)
    half_idx = len(x) // 2
    
    # --- WAVEFORM LOGIC ---
    if wave_type == "Square Wave (0 to 1)":
        # THIS MATCHES YOUR MATLAB QUESTION
        # The signal goes from 0 to 1.
        f[:] = 1
        f[:half_idx] = 0 
    elif wave_type == "Square Wave (-1 to 1)":
        # THIS IS THE STANDARD PHYSICS WAVE
        # The signal goes from -1 to 1.
        f[:half_idx] = -1
        f[half_idx:] = 1
    elif wave_type == "Sawtooth Wave":
        f = np.linspace(-1, 1, len(x))
    elif wave_type == "Triangle Wave":
        p1 = np.linspace(-1, 1, half_idx)
        p2 = np.linspace(1, -1, len(x) - half_idx)
        f = np.concatenate((p1, p2))

    # 3. Calculate Fourier Coefficients (Vectorized)
    # A0 (Average Value)
    # Integral of f(x) over the period, divided by L
    A0 = (1/L) * np.sum(f) * dx

    # Harmonics Setup
    # n_vec is a column vector: [[1], [2], ... [N]]
    n_vec = np.arange(1, N + 1).reshape(-1, 1)
    
    # Calculate angles for all harmonics and all time points at once
    # Formula: 2*pi*n*x / L
    args = (2 * np.pi * n_vec * x) / L
    
    # Numerical Integration for An and Bn
    # Summing across axis=1 performs the integral for every harmonic instantly
    An = (2/L) * np.sum(f * np.cos(args), axis=1) * dx
    Bn = (2/L) * np.sum(f * np.sin(args), axis=1) * dx
    
    # 4. Reconstruction (Synthesis)
    # Summing up all the sine and cosine terms
    terms = (An.reshape(-1,1) * np.cos(args)) + (Bn.reshape(-1,1) * np.sin(args))
    fs = A0 + np.sum(terms, axis=0)
    
    # 5. Analysis (RMS and Gibbs)
    rms_target = np.sqrt(np.mean(f**2))
    rms_approx = np.sqrt(np.mean(fs**2))
    rms_error = np.sqrt(np.mean((f - fs)**2))
    
    # Gibbs Phenomenon check
    max_overshoot = np.max(np.abs(fs))
    target_max = np.max(np.abs(f))
    gibbs_percentage = 0
    if target_max > 0:
        gibbs_percentage = ((max_overshoot - target_max) / target_max) * 100
    
    # Spectrum Magnitude for plotting frequency domain
    magnitudes = np.sqrt(An**2 + Bn**2)
    
    return x, f, fs, n_vec.flatten(), magnitudes, rms_target, rms_approx, rms_error, gibbs_percentage

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Parameters")
    # This dropdown allows you to switch between the "Exam Mode" (0 to 1) 
    # and "Standard Mode" (-1 to 1)
    wave_type = st.selectbox("Select Waveform", 
                             ["Square Wave (0 to 1)", "Square Wave (-1 to 1)", "Sawtooth Wave", "Triangle Wave"])
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        # Default L=1.0 matches your MATLAB notes
        L = st.number_input("Period (L)", value=1.0, step=0.1)
    with col2:
        # Default dx=0.005 matches your MATLAB notes
        dx = st.number_input("Step Size (dx)", value=0.005, format="%.3f")
        
    N = st.slider("Number of Harmonics (N)", 1, 1000, 50)
    st.write(f"Total Points: {int(L/dx)}")

# --- RUN CALCULATION ---
try:
    x, f, fs, n_vec, mags, rms_t, rms_a, rms_e, gibbs_p = calculate_fourier(wave_type, L, dx, N)
except Exception as e:
    st.error(f"Calculation Error: {e}")
    st.stop()

# --- DISPLAY RESULTS ---
tab1, tab2, tab3 = st.tabs(["Time Domain", "Frequency Spectrum", "Error Analysis"])

with tab1:
    st.subheader("Signal Reconstruction")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Target RMS", f"{rms_t:.4f}")
    c2.metric("Approx RMS", f"{rms_a:.4f}")
    c3.metric("RMS Error", f"{rms_e:.4f}")

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    # Plot target in black dashed
    ax1.plot(x, f, 'k--', linewidth=1.5, label="Target f(x)", alpha=0.5)
    # Plot approximation in red
    ax1.plot(x, fs, color='#FF4B4B', linewidth=1.5, label=f"Fourier (N={N})")
    ax1.set_xlim(0, L)
    ax1.set_xlabel("Time (x)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig1, use_container_width=True)

with tab2:
    st.subheader("Frequency Spectrum")
    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    
    # Limit bars shown if N is very large so the chart is readable
    limit = min(N, 50)
    
    ax2.bar(n_vec[:limit], mags[:limit], color='#0083B8', width=0.6)
    ax2.set_xlabel("Harmonic Index (n)")
    ax2.set_ylabel("Magnitude (Coefficient Strength)")
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig2, use_container_width=True)

with tab3:
    st.subheader("Error Analysis")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    
    error = f - fs
    # Handle NaN values safely
    error = np.nan_to_num(error, nan=0.0)
    
    ax3.plot(x, error, color='purple', linewidth=1)
    ax3.fill_between(x, error, 0, color='purple', alpha=0.1)
    ax3.set_xlim(0, L)
    ax3.set_title("Approximation Error (f(x) - Fourier(x))")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3, use_container_width=True)

# Mathematical Formulas Reference
with st.expander("Formulas Used"):
    st.write("Calculations are performed using the general period $L$:")
    st.latex(r"A_0 = \frac{1}{L} \int_0^L f(x) \, dx")
    st.latex(r"A_n = \frac{2}{L} \int_0^L f(x) \cos\left(\frac{2\pi n x}{L}\right) \, dx")
    st.latex(r"B_n = \frac{2}{L} \int_0^L f(x) \sin\left(\frac{2\pi n x}{L}\right) \, dx")
