import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Black Hole Merger Viewer")

t = st.slider("Select time", 0.0, 1.0, 0.1)

x = np.linspace(0, 10, 100)
y = np.sin(x + t)

fig, ax = plt.subplots()
ax.plot(x, y)

st.pyplot(fig)