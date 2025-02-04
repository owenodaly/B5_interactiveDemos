import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import scipy
#st.info(f"SciPy version: {scipy.__version__}")
#res = stats.goodness_of_fit(stats.norm, pseudo_data, known_params=known_params,
#                                    statistic='ks', random_state=rng)
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_bold_signal(time, freq=0.1, noise_level=0.1):
    """Generate underlying BOLD signal."""
    signal = (np.sin(2 * np.pi * freq * time) + 
             0.5 * np.sin(2 * np.pi * 0.05 * time) +
             0.3 * np.sin(2 * np.pi * 0.15 * time))
    return signal

def sample_bold_signal(bold_signal, time, slice_times, tr, noise_level=0.1):
    """Sample BOLD signal at specific slice times with noise."""
    n_timepoints = len(time)
    n_slices = len(slice_times)
    sampled_data = np.zeros((n_slices, n_timepoints))
    
    # Generate continuous time points for interpolation
    continuous_time = np.linspace(0, time[-1], 1000)
    continuous_bold = generate_bold_signal(continuous_time)
    bold_interpolator = interp1d(continuous_time, continuous_bold, 
                                kind='cubic', bounds_error=False, 
                                fill_value="extrapolate")
    
    for slice_idx in range(n_slices):
        # Calculate actual sampling times for this slice
        sampling_times = time + slice_times[slice_idx]
        
        # Sample the BOLD signal at these times
        sampled_signal = bold_interpolator(sampling_times)
        
        # Add slice-specific noise
        noise = np.random.normal(0, noise_level, size=n_timepoints)
        sampled_data[slice_idx] = sampled_signal + noise
    
    return sampled_data

def get_slice_acquisition_times(n_slices, tr, ta, slice_order='ascending'):
    """Calculate acquisition time for each slice."""
    if slice_order == 'interleaved':
        order = np.concatenate([np.arange(0, n_slices, 2), 
                              np.arange(1, n_slices, 2)])
    else:
        order = np.arange(n_slices)
    
    slice_times = (ta / n_slices) * np.argsort(order)
    return slice_times

def slice_timing_correction(data, slice_times, tr, interpolation='linear'):
    """Apply slice timing correction using specified interpolation method."""
    n_slices, n_timepoints = data.shape
    time_points = np.arange(n_timepoints) * tr
    corrected_data = np.zeros_like(data)
    
    for slice_idx in range(n_slices):
        orig_times = time_points + slice_times[slice_idx]
        target_times = time_points + tr/2
        
        if interpolation == 'linear':
            interpolator = interp1d(orig_times, data[slice_idx], 
                                  kind='linear', bounds_error=False, 
                                  fill_value="extrapolate")
        else:
            interpolator = interp1d(orig_times, data[slice_idx], 
                                  kind='cubic', bounds_error=False, 
                                  fill_value="extrapolate")
            
        corrected_data[slice_idx] = interpolator(target_times)
    
    return corrected_data

def plot_timeseries(original_data, corrected_data, time, slice_times, tr):
    """Create interactive plot comparing original and corrected data."""
    # Generate underlying BOLD signal for reference
    continuous_time = np.linspace(0, time[-1], 1000)
    true_bold = generate_bold_signal(continuous_time)
    
    fig = make_subplots(rows=3, cols=1, 
                       subplot_titles=('True BOLD Signal', 
                                     'Sampled Data', 
                                     'Corrected Data'))
    
    # Plot true BOLD signal
    fig.add_trace(
        go.Scatter(x=continuous_time, y=true_bold, 
                  name='True BOLD', line=dict(color='black')),
        row=1, col=1
    )
    
    # Plot sampled data
    for i in range(len(original_data)):
        fig.add_trace(
            go.Scatter(x=time, y=original_data[i], 
                      name=f'Slice {i}', mode='markers+lines'),
            row=2, col=1
        )
        
    # Plot corrected data
    for i in range(len(corrected_data)):
        fig.add_trace(
            go.Scatter(x=time, y=corrected_data[i], 
                      name=f'Slice {i} (corrected)', 
                      mode='markers+lines'),
            row=3, col=1
        )
    
    fig.update_layout(height=1000, title_text="Slice Timing Correction Demo")
    return fig

# Streamlit UI
st.title('fMRI Slice Timing Correction Demo')

col1, col2, col3 = st.columns(3)
with col1:
    n_timepoints = st.slider('Number of Timepoints', 20, 100, 50)
    n_slices = st.slider('Number of Slices', 2, 10, 3)
    
with col2:
    tr = st.slider('TR (s)', 0.5, 5.0, 2.0, 0.1)
    ta = st.slider('TA (s)', 0.5, 5.0, 1.8, 0.1)

with col3:
    slice_order = st.selectbox('Slice Order', 
                             ['ascending', 'interleaved'])
    interpolation = st.selectbox('Interpolation Method', 
                               ['linear', 'sinc'])

# Generate and process data
slice_times = get_slice_acquisition_times(n_slices, tr, ta, slice_order)
time = np.arange(n_timepoints) * tr

# Generate true BOLD and sample it
data = sample_bold_signal(None, time, slice_times, tr)
corrected_data = slice_timing_correction(data, slice_times, tr, interpolation)

# Plotting
fig = plot_timeseries(data, corrected_data, time, slice_times, tr)
st.plotly_chart(fig, use_container_width=True)

# Display slice acquisition times
st.subheader('Slice Acquisition Times')
times_df = pd.DataFrame({
    'Slice': range(n_slices),
    'Acquisition Time (s)': slice_times
})
st.dataframe(times_df)
