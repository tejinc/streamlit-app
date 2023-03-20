import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

##############################################################################
# Workaround for the limited multi-threading support in matplotlib.
# Per the docs, we will avoid using `matplotlib.pyplot` for figures:
# https://matplotlib.org/3.3.2/faq/howto_faq.html#how-to-use-matplotlib-in-a-web-application-server.
# Moreover, we will guard all operations on the figure instances by the
# class-level lock in the Agg backend.
##############################################################################
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock



# utility functions
def drawPlot( df, x, y, ptype="None"):
    if ptype == "None": 
        return

    fig, ax = plt.subplots()

    if (x == "None") ^ (y == "None"):
        var = x if y == "None"  else y
        ax.hist( df[var])
    elif (x != "None" ) and (y !="None"):
        if ptype == "Scatter":
            ax.scatter( df[x],df[y])
        elif ptype == "Histogram":
            ax.hist2d(df[x],df[y])
    else:
        return
    with _lock:
        st.pyplot(fig)

