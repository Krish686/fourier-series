import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set up the page layout
st.set_page_config(page_title="Fourier Series Experiment", layout="wide")

st.title("Fourier Series Simulation")
st.write("Visualizing Fourier Series approximation, RMS values, and the Gibbs Phenomenon.")

# Function to perform the numerical calculations
@st.cache_data(show_spinner=False)
def calculate_fourier(wave_type, L, dx, N):
    # Create the x array (domain 0 to L)
    # Adding a small epsilon to ensure the last point L is included
    x = np.arange(dx, L + dx/100000, dx)
    
    # Initialize the target function f(x)
    f = np.zeros_like(x)
    half_idx = len(x) // 2
    
    # Waveform logic based on selection
    if wave_type == "Square Wave":
        f[:half_idx] = -1 
        f[half_idx:] = 1 
    elif wave_type == "Sawtooth Wave":
        f = np.linspace(-1, 1, len(x))
    elif wave_type == "Triangle Wave":
        p1 = np.linspace(-1, 1, half_idx)
        p2 = np.linspace(1, -1, len(x) - half_idx)
        f = np.concatenate((p1, p2))

    # Calculate A0 (Average value)
    A0 = (1/L) * np.sum(f) * dx

    # Vectorized calculation for harmonics (An and Bn)
    # Using matrix math instead of loops for speed
    n_vec = np.arange(1, N + 1).reshape(-1, 1)
    args = (2 * np.pi * n_vec * x) / L
    
    # Numerical Integration
    An = (2/L) * np.sum(f * np.cos(args), axis=1) * dx
    Bn = (2/L) * np.sum(f * np.sin(args), axis=1) * dx
    
    # Reconstruct the signal (Synthesis)
    terms = (An.reshape(-1,1) * np.cos(args)) + (Bn.reshape(-1,1) * np.sin(args))
    fs = A0 + np.sum(terms, axis=0)
    
    # RMS Calculations
    rms_target = np.sqrt(np.mean(f**2))
    rms_approx = np.sqrt(np.mean(fs**2))
    rms_error = np.sqrt(np.mean((f - fs)**2))
    
    # Gibbs Phenomenon check
    max_overshoot = np.max(np.abs(fs))
    target_max = np.max(np.abs(f))
    gibbs_percentage = 0
    if target_max > 0:
        gibbs_percentage = ((max_overshoot - target_max) / target_max) * 100
    
    # Spectrum Data
    magnitudes = np.sqrt(An**2 + Bn**2)
    
    return x, f, fs, n_vec.flatten(), magnitudes, rms_target, rms_approx, rms_error, gibbs_percentage

# Sidebar for user inputs
with st.sidebar:
    st.header("Parameters")
    wave_type = st.selectbox("Select Waveform", ["Square Wave", "Sawtooth Wave", "Triangle Wave"])
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        L = st.number_input("Period (L)", value=1.0, step=0.1)
    with col2:
        dx = st.number_input("Step Size (dx)", value=0.005, format="%.3f")
        
    N = st.slider("Number of Harmonics (N)", 1, 1000, 50)
    st.write(f"Total Points: {int(L/dx)}")

# Run the calculation
try:
    x, f, fs, n_vec, mags, rms_t, rms_a, rms_e, gibbs_p = calculate_fourier(wave_type, L, dx, N)
except Exception as e:
    st.error(f"Error in calculation: {e}")
    st.stop()

# Display Results
tab1, tab2, tab3 = st.tabs(["Time Domain & RMS", "Frequency Spectrum", "Error Analysis"])

with tab1:
    st.subheader("Signal Reconstruction")
    
    # Display RMS values
    c1, c2, c3 = st.columns(3)
    c1.metric("Target RMS", f"{rms_t:.4f}")
    c2.metric("Approx RMS", f"{rms_a:.4f}")
    c3.metric("RMS Error", f"{rms_e:.4f}")

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(x, f, 'k--', linewidth=1.5, label="Original f(x)", alpha=0.6)
    ax1.plot(x, fs, color='#FF4B4B', linewidth=1.5, label=f"Fourier Approximation (N={N})")
    ax1.set_xlim(0, L)
    ax1.set_xlabel("x")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)
    
    if abs(rms_t - rms_a) < 0.01:
        st.info("RMS values have converged.")

with tab2:
    col_gibbs, col_spec = st.columns([1, 2])
    
    with col_gibbs:
        st.subheader("Gibbs Phenomenon")
        st.metric("Overshoot %", f"{gibbs_p:.2f}%")
        st.write("For square waves, this stays around 9%.")
        
    with col_spec:
        st.subheader("Frequency Spectrum")
        fig2, ax2 = plt.subplots(figsize=(10, 3.5))
        # Limit bars shown if N is very large
        limit = min(N, 50)
        ax2.bar(n_vec[:limit], mags[:limit], color='#0083B8', width=0.6)
        ax2.set_xlabel("Harmonic (n)")
        ax2.set_ylabel("Magnitude")
        ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

with tab3:
    st.subheader("Deviation Analysis")
    
    # Calculate error
    raw_error = f - fs
    # Handle any potential infinity/nan values to prevent plot crashes
    error = np.nan_to_num(raw_error, nan=0.0, posinf=0.0, neginf=0.0)
    mse = np.mean(error**2)
    
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(x, error, color='purple', linewidth=1)
    ax3.fill_between(x, error, 0, color='purple', alpha=0.1)
    
    # Set Y-limits to avoid rendering issues with spikes
    y_min, y_max = np.min(error), np.max(error)
    span = y_max - y_min
    if span == 0: span = 1.0
    ax3.set_ylim(y_min - 0.1*span, y_max + 0.1*span)
    ax3.set_xlim(0, L)
    
    # Attempt to annotate the Gibbs spike
    try:
        max_idx = np.argmax(np.abs(error))
        x_peak = float(x[max_idx])
        y_peak = float(error[max_idx])
        
        # Only draw arrow if peak is within view
        if y_min <= y_peak <= y_max:
            ax3.annotate('Gibbs Spike', xy=(x_peak, y_peak), 
                         xytext=(x_peak, y_peak + (0.2 * span * (1 if y_peak>0 else -1))),
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         ha='center')
    except:
        pass 
    
    ax3.set_title(f"Error Signal (MSE: {mse:.5f})")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

# Mathematical Formulas Reference
with st.expander("Formulas Used"):
    st.write("Coefficients calculated via Numerical Integration:")
    st.latex(r"A_n = \frac{2}{L} \sum f(x) \cos\left(\frac{2\pi n x}{L}\right) dx")
    st.latex(r"B_n = \frac{2}{L} \sum f(x) \sin\left(\frac{2\pi n x}{L}\right) dx")
    st.latex(r"RMS = \sqrt{\frac{1}{L} \int f(x)^2 dx}")