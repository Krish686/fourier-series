# Fourier Series Simulation 🌊

A high-performance, interactive web application to visualize the **Fourier Series decomposition** of various periodic waveforms. Built with **Python**, **Streamlit**, and **NumPy**, this tool uses numerical integration to compute Fourier coefficients and reconstruct signals in real-time.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Mathematical Background](#mathematical-background)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)

## Overview
This application demonstrates how complex periodic functions can be constructed by summing simple sine and cosine waves (harmonics). It allows users to adjust the **period (L)**, **resolution (dx)**, and **number of harmonics (N)** to instantly see the effect on the signal reconstruction and the resulting Gibbs phenomenon.

The calculation engine is fully **vectorized** using NumPy, allowing it to compute thousands of harmonics over high-resolution domains instantly.

## Features

### 🎛️ Interactive Controls
* **Waveform Selection:** Choose between:
    * *Square Wave (0 to 1):* Unipolar pulse.
    * *Square Wave (-1 to 1):* Bipolar standard physics wave.
    * *Sawtooth Wave.*
    * *Triangle Wave.*
* **Parameter Tuning:**
    * Adjust Period ($L$).
    * Adjust Step Size ($dx$) for numerical precision.
    * Slider for Harmonics ($N$) from 1 to 1000.

### 📊 Visualizations
The app provides three distinct analysis tabs:
1.  **Time Domain:** Overlays the target function $f(x)$ (black dashed) with the Fourier approximation (red).
2.  **Frequency Spectrum:** Bar chart showing the magnitude ($\sqrt{A_n^2 + B_n^2}$) of specific harmonics.
3.  **Error Analysis:** Visualizes the difference $f(x) - f_{approx}(x)$ and calculates RMS error.

### 📉 Metrics
* **RMS Target vs. Approx:** Compares the Root Mean Square energy of the original vs. reconstructed signal.
* **Gibbs Phenomenon:** Calculates the percentage overshoot at discontinuities.

## Mathematical Background
The application performs **Numerical Integration** (Riemann Sums) to find the Fourier coefficients. For a function $f(x)$ defined over interval $[0, L]$:

**1. DC Component ($A_0$):**
$$A_0 = \frac{1}{L} \int_0^L f(x) \, dx$$

**2. Cosine Coefficients ($A_n$):**
$$A_n = \frac{2}{L} \int_0^L f(x) \cos\left(\frac{2\pi n x}{L}\right) \, dx$$

**3. Sine Coefficients ($B_n$):**
$$B_n = \frac{2}{L} \int_0^L f(x) \sin\left(\frac{2\pi n x}{L}\right) \, dx$$

**4. Reconstruction:**
$$f(x) \approx A_0 + \sum_{n=1}^{N} \left[ A_n \cos\left(\frac{2\pi n x}{L}\right) + B_n \sin\left(\frac{2\pi n x}{L}\right) \right]$$

*Note: The app calculates these integrals discretely using the step size `dx`.*

## Installation

1.  **Clone the repository** (or download the files):
    ```bash
    git clone <your-repo-url>
    cd fourier-series-sim
    ```

2.  **Create a virtual environment** (Optional but recommended):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit application from your terminal:

```bash
streamlit run app.py
