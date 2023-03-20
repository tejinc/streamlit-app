import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from streamlit import session_state as ss


#local import
#import utils.load as loader
from utils import load, setting







setting.initialize_setting()

st.set_page_config(
    page_title="Data Analyser",
    page_icon="📊",
)

#initialize session state to store
# 1. input date state and input data 
#    the state is false before loading data
#    the state becomes true when data is loaded and stored in a state
#    the state becomes false again if the stored data is none

 
with st.sidebar:
    load.load_file()
    st.subheader("Settings")

    #file uploader
    # uploader_label = "Please upload csv or xslx files"
    # st.session_state["input_file"] = st.file_uploader( uploader_label, type = loader.get_types() )

st.markdown(
    """
    This is a demo app to perform simple data exploration and 
    train pre-defined models.

    **👈 Select a page from the sidebar** to check out its 
    functionalities!
    ### Functionalities presently available
    - Make exploratory plots 
    - Train with various parameters 
"""
)

