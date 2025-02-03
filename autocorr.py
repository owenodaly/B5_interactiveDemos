import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Helper functions
# -----------------------------

def generate_optimized_ts(N=200, rho=0.8, sigma=1.0, seed=42):
    """Generate an AR(1) process with a superimposed sine wave (optimized to illustrate autocorrelation)."""
    np.random.seed(seed)
    x = np.zeros(N)
    x[0] = np.random.normal(0, sigma)
    for t in range(1, N):
        x[t] = rho * x[t-1] + np.random.normal(0, sigma)
    # add a sine wave overlay
    t_vals = np.linspace(0, 4*np.pi, N)
    x += 0.5 * np.sin(t_vals)
    return x

def generate_fmri_ts(N=200, rho=0.6, sigma=1.0, seed=42):
    """Generate an AR(1) process that mimics fMRI signal characteristics (including drift)."""
    np.random.seed(seed)
    x = np.zeros(N)
    x[0] = np.random.normal(0, sigma)
    for t in range(1, N):
        x[t] = rho * x[t-1] + np.random.normal(0, sigma)
    # add a low-frequency drift (e.g., linear drift)
    drift = np.linspace(0, 1, N)
    x += drift
    return x

def apply_lag(x, lag):
    """
    Return a lagged version of x.
    For positive lag, the lagged series is delayed (i.e. its beginning is NaN).
    For negative lag, the series is advanced.
    """
    x = np.asarray(x)
    x_lag = np.empty_like(x)
    if lag >= 0:
        x_lag[:lag] = np.nan
        if lag > 0:
            x_lag[lag:] = x[:-lag]
        else:
            x_lag = x.copy()
    else:
        lag = abs(lag)
        x_lag[-lag:] = np.nan
        x_lag[:-lag] = x[lag:]
    return x_lag

def create_heatmap_data(x, n_rows=10):
    """
    Create a 2D array by repeating the timeseries data to be used as a heatmap.
    """
    return np.tile(x, (n_rows, 1))

def shift_heatmap(img, shift):
    """
    Shift the heatmap horizontally. The shift value (in number of columns)
    can be positive (shift right) or negative (shift left). Areas moved in are set to NaN.
    """
    n_rows, n_cols = img.shape
    shifted = np.empty_like(img)
    shifted[:] = np.nan
    if shift >= 0:
        shifted[:, shift:] = img[:, :n_cols-shift]
    else:
        shift = abs(shift)
        shifted[:, :n_cols-shift] = img[:, shift:]
    return shifted

def compute_acf(x, max_lag=20):
    """
    Compute the autocorrelation function (ACF) for lags 0 to max_lag.
    """
    x = np.asarray(x)
    n = len(x)
    acf_vals = []
    x_mean = np.mean(x)
    var = np.var(x)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf_vals.append(1.0)
        else:
            # compute covariance at this lag
            cov = np.sum((x[:-lag] - x_mean) * (x[lag:] - x_mean)) / (n - lag)
            acf_vals.append(cov / var)
    return np.array(acf_vals)

# -----------------------------
# Streamlit App Layout
# -----------------------------

st.title("Interactive Tutorial: Temporal Autocorrelation in fMRI Data")
st.markdown("""
This interactive app demonstrates temporal autocorrelation in timeseries data (e.g., fMRI signals)
and shows how the lag between measurements is reflected in the covariance structure and the autocorrelation function (ACF).

**Panels:**
1. **Timeseries & Lagged Timeseries:** Line plots of the original and lagged timeseries.
2. **Heatmap Visualization:** Grey-scale heatmaps of the timeseries where a slider simulates a draggable lag.
3. **Covariance Matrix & Autocorrelation Function:** A theoretical AR(1) covariance matrix with highlighted lag elements and the empirical ACF plot.

Use the sidebar to adjust settings.
""")

# Sidebar inputs
st.sidebar.header("Tutorial Settings")
timeseries_type = st.sidebar.radio("Select Timeseries Type", ("Optimized", "fMRI-like"))
lag_value = st.sidebar.slider("Lag (in timepoints)", -20, 20, 0, step=1)
N = st.sidebar.number_input("Number of Timepoints", min_value=50, max_value=500, value=200, step=10)

# Generate synthetic data
if timeseries_type == "Optimized":
    x = generate_optimized_ts(N=N)
    true_rho = 0.8
    true_sigma2 = 1.0
else:
    x = generate_fmri_ts(N=N)
    true_rho = 0.6
    true_sigma2 = 1.0

# Compute lagged version for line plots (non-overlapping parts as NaN)
x_lag = apply_lag(x, lag_value)

# -----------------------------
# Panel 1: Timeseries Line Plots
# -----------------------------
st.subheader("Panel 1: Timeseries and Lagged Timeseries (Line Plots)")

fig1, ax1 = plt.subplots(figsize=(10, 4))
time = np.arange(N)
ax1.plot(time, x, label="Original Timeseries", color="C0", lw=2)
ax1.plot(time, x_lag, label=f"Lagged Timeseries (lag = {lag_value})", color="C1", lw=2, ls="--")
ax1.set_xlabel("Timepoints")
ax1.set_ylabel("Signal Amplitude")
ax1.legend()
ax1.set_title("Line Plots of Original and Lagged Timeseries")
st.pyplot(fig1)

# -----------------------------
# Panel 2: Grey-scale Heatmaps
# -----------------------------
st.subheader("Panel 2: Grey-scale Heatmap Visualization")

# Create heatmap data by repeating the 1D timeseries
n_rows = 10
heatmap_orig = create_heatmap_data(x, n_rows=n_rows)
heatmap_lag = shift_heatmap(heatmap_orig, lag_value)

fig2, axes2 = plt.subplots(2, 1, figsize=(10, 6))
im1 = axes2[0].imshow(heatmap_orig, aspect="auto", cmap="gray", interpolation="none")
axes2[0].set_title("Original Timeseries (Heatmap)")
axes2[0].set_ylabel("Replications")
plt.colorbar(im1, ax=axes2[0], orientation="horizontal")

im2 = axes2[1].imshow(heatmap_lag, aspect="auto", cmap="gray", interpolation="none")
axes2[1].set_title(f"Lagged Timeseries (Heatmap) shifted by {lag_value} timepoints")
axes2[1].set_xlabel("Timepoints")
axes2[1].set_ylabel("Replications")
plt.colorbar(im2, ax=axes2[1], orientation="horizontal")

st.pyplot(fig2)

st.markdown("""
*Note:* The horizontal slider in the sidebar simulates a draggable lag adjustment for the heatmap.
""")

# -----------------------------
# Panel 3: Covariance Matrix & Autocorrelation Function
# -----------------------------
st.subheader("Panel 3: Covariance Matrix Highlighting Lag and Autocorrelation Function")

# Use a theoretical AR(1) covariance matrix computed from the known parameters.
subset_N = min(50, N)  # display a subset for clarity
indices = np.arange(subset_N)
diff = np.abs(np.subtract.outer(indices, indices))
V_theoretical = true_sigma2 * (true_rho ** diff)

# Determine lag to highlight (using absolute lag since the autocorrelation is symmetric)
lag_to_highlight = abs(lag_value)
if lag_to_highlight < subset_N:
    indices_high = np.arange(subset_N - lag_to_highlight)
    x_coords = indices_high + lag_to_highlight
    y_coords = indices_high
else:
    x_coords = np.array([])
    y_coords = np.array([])

# Create two columns: one for the covariance matrix and one for the ACF plot.
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Theoretical AR(1) Covariance Matrix**")
    fig_cov, ax_cov = plt.subplots(figsize=(6,5))
    im_cov = ax_cov.imshow(V_theoretical, cmap="viridis", origin="upper")
    if len(x_coords) > 0:
        # Highlight the elements corresponding to the selected lag on both sides of the diagonal.
        ax_cov.scatter(x_coords, y_coords, facecolors='none', edgecolors='red', s=100, label=f"Lag = {lag_value}")
        if lag_to_highlight > 0:
            ax_cov.scatter(y_coords, x_coords, facecolors='none', edgecolors='red', s=100)
    ax_cov.set_title("AR(1) Covariance Matrix\nwith Highlighted Lag")
    ax_cov.set_xlabel("Timepoints")
    ax_cov.set_ylabel("Timepoints")
    ax_cov.legend(loc='upper right')
    plt.colorbar(im_cov, ax=ax_cov)
    st.pyplot(fig_cov)

with col2:
    st.markdown("**Autocorrelation Function (ACF)**")
    max_lag = 20
    acf_vals = compute_acf(x, max_lag=max_lag)
    fig_acf, ax_acf = plt.subplots(figsize=(6,4))
    markerline, stemlines, baseline = ax_acf.stem(range(len(acf_vals)), acf_vals, basefmt=" ")
    # Highlight the autocorrelation value at the selected lag if within range
    if lag_to_highlight <= max_lag:
        ax_acf.plot(lag_to_highlight, acf_vals[lag_to_highlight], 'ro', markersize=10, label=f"ACF at lag = {lag_to_highlight}")
    ax_acf.set_xlabel("Lag")
    ax_acf.set_ylabel("Autocorrelation")
    ax_acf.set_title("Empirical Autocorrelation Function")
    ax_acf.legend()
    st.pyplot(fig_acf)

st.markdown("""
### Explanation
- **Covariance Matrix:**  
  For an AR(1) process, the theoretical covariance between observations at times \(t\) and \(t+d\) is:  
  \[
  \mathrm{Cov}(x_t, x_{t+d}) = \sigma^2 \rho^{|d|}
  \]  
  The left panel shows this covariance matrix for a subset of timepoints with the elements corresponding to the selected lag highlighted in red.

- **Autocorrelation Function (ACF):**  
  The ACF plot (right panel) displays the empirical correlation between the timeseries values and their lagged versions for lags \(0\) to \(20\).  
  The red marker highlights the autocorrelation at the (absolute) selected lag.

Together, these visualizations help illustrate how lag influences both the covariance structure and the autocorrelation of the timeseries.
""")
