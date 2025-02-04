import streamlit as st
import timing  # Import your tutorials
import autocorr
import smooth
import predbold
from scipy import stats

# Landing page
st.title("Interactive Tutorials")

# Select tutorial
tutorial_choice = st.selectbox("Choose a tutorial:", ['BOLD', 'Autocorrelation', 'Smoothing','Slice-Timing Correction'])

if tutorial_choice == 'BOLD':
    predbold.run()  # Assume each tutorial script has a `run` function
elif tutorial_choice == 'Autocorrelation':
    autocorr.run()
elif tutorial_choice == 'Smoothing':
    smooth.run()
elif tutorial_choice == 'Slice-Timing Correction':
    timing.run()
