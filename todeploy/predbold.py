import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.stats import gamma
import seaborn as sns

def canonical_hrf(tr=1.5, duration=32.0):
    """Canonical HRF using the double-gamma function."""
    dt = tr
    time = np.arange(0, duration, dt)
    peak1 = 6
    under1 = 16
    ratio = 1/6
    hrf = gamma.pdf(time, peak1 / dt, scale=dt) - ratio * gamma.pdf(time, under1 / dt, scale=dt)
    hrf /= np.max(hrf)
    return hrf

def create_boxcar(onsets, durations, total_time, tr=1.0):
    """Create a boxcar function based on onsets and durations."""
    time = np.arange(0, total_time, tr)
    boxcar = np.zeros_like(time)
    for onset, duration in zip(onsets, durations):
        onset_idx = int(np.floor(onset / tr))
        duration_idx = int(np.ceil(duration / tr))
        if onset_idx < len(boxcar):
            boxcar[onset_idx:onset_idx + duration_idx] = 1
    return time, boxcar

def compute_correlation_matrix(timeseries_list):
    """Compute correlation matrix between timeseries."""
    n = len(timeseries_list)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr_matrix[i, j] = np.corrcoef(timeseries_list[i], timeseries_list[j])[0, 1]
    return corr_matrix

def compute_fft(signal, tr):
    """Compute FFT of the signal."""
    n = len(signal)
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(n, tr)
    
    # Get positive frequencies only
    pos_freq_mask = frequencies >= 0
    frequencies = frequencies[pos_freq_mask]
    fft_magnitude = np.abs(fft_result[pos_freq_mask])
    
    # Normalize magnitude
    fft_magnitude = fft_magnitude / np.max(fft_magnitude)
    
    return frequencies, fft_magnitude

def parse_input_string(input_str):
    """Parse input string to list of floats."""
    try:
        return [float(item.strip()) for item in input_str.split(",")]
    except ValueError:
        return None

def main():
    st.title("Multi-condition BOLD Timeseries Analysis")
    st.write("""
    This dashboard allows you to compare multiple experimental conditions and analyze their relationships 
    through timeseries correlation and frequency domain analysis.
    """)

    st.sidebar.header("Input Parameters")

    # Global parameters
    tr = st.sidebar.number_input("Repetition Time (TR) in seconds", min_value=0.1, value=1.0, step=0.1)
    num_conditions = st.sidebar.number_input("Number of Conditions", min_value=1, max_value=5, value=2)
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    all_onsets = []
    all_durations = []
    condition_names = []
    valid_inputs = True

    for i in range(num_conditions):
        st.sidebar.subheader(f"Condition {i+1}")
        name = st.sidebar.text_input(f"Condition {i+1} Name", f"Condition {i+1}")
        condition_names.append(name)
        
        onsets_input = st.sidebar.text_input(
            f"Onsets for {name} (seconds, comma-separated)", 
            value=f"{i*30}, {i*30 + 60}, {i*30 + 120}"
        )
        durations_input = st.sidebar.text_input(
            f"Durations for {name} (seconds, comma-separated)", 
            value="20, 20, 20"
        )
        
        onsets = parse_input_string(onsets_input)
        durations = parse_input_string(durations_input)
        
        if onsets is None or durations is None:
            st.error(f"Please enter valid numbers for {name}'s onsets and durations.")
            valid_inputs = False
            break
            
        if len(onsets) != len(durations):
            st.error(f"The number of onsets and durations must be the same for {name}.")
            valid_inputs = False
            break
            
        all_onsets.append(onsets)
        all_durations.append(durations)

    if valid_inputs:
        max_times = [max(onsets) + max(durations) for onsets, durations in zip(all_onsets, all_durations)]
        total_time = max(max_times) + 32
        hrf = canonical_hrf(tr=tr)

        # Create main figure for timeseries plots
        fig1 = plt.figure(figsize=(15, 8))
        gs1 = plt.GridSpec(2, 1, height_ratios=[1, 1.5])
        
        # Box-car plot
        ax1 = fig1.add_subplot(gs1[0])
        # BOLD response plot
        ax2 = fig1.add_subplot(gs1[1])

        # Lists to store BOLD responses for analysis
        bold_responses = []
        
        # Plot box-car functions and BOLD responses
        for i in range(num_conditions):
            time, boxcar = create_boxcar(all_onsets[i], all_durations[i], total_time, tr=tr)
            ax1.step(time, boxcar + i*0.2, where='post', 
                    label=condition_names[i], color=colors[i], alpha=0.7)
            
            bold = convolve(boxcar, hrf)[:len(boxcar)]
            if np.max(bold) != 0:
                bold /= np.max(bold)
            ax2.plot(time, bold, label=condition_names[i], color=colors[i])
            bold_responses.append(bold)

        # Customize time-domain plots
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Box-Car Functions')
        ax1.legend()
        ax1.grid(True)

        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Normalized Amplitude')
        ax2.set_title('Predicted BOLD Timeseries')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        st.pyplot(fig1)

        # Create second figure for correlation and FFT
        fig2 = plt.figure(figsize=(15, 6))
        gs2 = plt.GridSpec(1, 2)
        
        # Correlation matrix plot
        ax3 = fig2.add_subplot(gs2[0])
        # FFT plot
        ax4 = fig2.add_subplot(gs2[1])

        # Plot correlation matrix
        corr_matrix = compute_correlation_matrix(bold_responses)
        sns.heatmap(corr_matrix, ax=ax3, cmap='RdBu_r', vmin=-1, vmax=1, 
                   xticklabels=condition_names, yticklabels=condition_names, 
                   annot=True, fmt='.2f', square=True, cbar_kws={'label': 'Correlation'})
        ax3.set_title('Timeseries Correlation Matrix')

        # Plot FFT
        for i, bold in enumerate(bold_responses):
            frequencies, fft_magnitude = compute_fft(bold, tr)
            # Only plot up to 0.15 Hz (typical BOLD range) - changed to 0.05Hz by Owen
            mask = frequencies <= 0.05
            ax4.plot(frequencies[mask], fft_magnitude[mask], 
                    label=condition_names[i], color=colors[i])
        
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Normalized FFT Magnitude')
        ax4.set_title('Frequency Spectrum (FFT)')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        st.pyplot(fig2)

        # Display analysis details
        st.subheader("Analysis Details")
        
        st.write("**Correlation Matrix:**")
        st.write("""
        The correlation matrix shows the Pearson correlation coefficient between each pair of predicted 
        BOLD timeseries. Values range from -1 (perfect negative correlation) to 1 (perfect positive correlation).
        """)
        
        st.write("**Frequency Spectrum:**")
        st.write("""
        The frequency spectrum shows the magnitude of different frequency components in each timeseries, 
        computed using the Fast Fourier Transform (FFT). The x-axis is limited to 0.15 Hz to focus on the 
        typical BOLD frequency range. The magnitude is normalized to better compare the relative 
        contribution of different frequencies across conditions.
        """)

        # Input details
        st.subheader("Input Details")
        for i in range(num_conditions):
            st.write(f"**{condition_names[i]}:**")
            st.write(f"- Onsets: {all_onsets[i]}")
            st.write(f"- Durations: {all_durations[i]}")
        
        st.write(f"**Repetition Time (TR):** {tr} seconds")

if __name__ == "__main__":
    main()