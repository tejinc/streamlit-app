import streamlit as st
import matplotlib.pyplot as plt
from sklearn import decomposition
import numpy as np

from streamlit import session_state as ss
import plotly.express as px
import streamlit.components.v1 as components
from utils import load, setting, train_xgboot, data_filter

##############################################################################
# Workaround for the limited multi-threading support in matplotlib.
# Per the docs, we will avoid using `matplotlib.pyplot` for figures:
# https://matplotlib.org/3.3.2/faq/howto_faq.html#how-to-use-matplotlib-in-a-web-application-server.
# Moreover, we will guard all operations on the figure instances by the
# class-level lock in the Agg backend.
##############################################################################
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock





setting.initialize_setting()

#title of the app
st.set_page_config(
    page_title="Data Explorer",
    page_icon="📊",
    layout="wide"
)

st.title("Data Explorer")

st.markdown(
    """
    ### Instruction
    This app takes a file, and encode each categorical data using an OrdinalEncoder.
    The user could then specify which data -- encoded or original -- to make simple plots. 
    The user could select up to 3 features for plotting, 
    - For one or two features, a user could make histogram
    - For two or three features, a user could make scatter plot

    
    The histogram and scatter plots will automatically update. The user could
    apply additional filtering using the control panel on the right. 
    - Numeric columns: select a range of data
    - Categorical columns: select categories to be kept, default to true for all categories.

    Finally, the user could also perform PCA on the formated data. 
"""
)




with st.sidebar:
    load.load_file()
    #subheader
    if ss["df"] is None:
        st.write("Please upload a data file.")

    if st.button("Format data",help="Encode categorical data into numbers", disabled=(ss["df"] is None)):
        train_xgboot.format_data()
    st.subheader("Explorer Setting")
    select_formated_df = st.checkbox("Use encoded data", disabled=(ss["formated_df"] is None))
    df = ss["formated_df"] if select_formated_df else ss["df"]
    select_numeric = st.sidebar.checkbox("Select only numeric columns", disabled = (df is None))
    options = []
    if df is not None:
        options = df.select_dtypes(include=np.number).columns.tolist() if select_numeric else list(df.columns)
    variables = st.sidebar.multiselect("Select up to 3 variables from list", options=options, max_selections=3)
    with st.expander("Plot options"):
        marker_size = st.select_slider("Marker size", value=1, options=np.arange(0,10,0.5))
        opacity = st.slider("Marker opacity", value=1., min_value=0., max_value=1.)


main_col1, main_col2 = st.columns([3,1])

with main_col2:
    # select color column:
    color_col = st.selectbox("Select color column:", options=["None"]+ list(df.columns))
    color_col = None if color_col == "None" else color_col
    df_filtered=data_filter.data_filter(df)

with main_col1:
    with st.expander("Make Histogram (1 or 2 D)", expanded =(len(variables) in [1,2])):
        sel_options_num = dict()
        sel_options_cat = dict()
        if len(variables) in [1,2]:
            fig = None
            if len(variables) == 1:
                x = variables[0]
                fig = px.histogram(df_filtered, x=x,color=color_col)
            elif len(variables) == 2:
                x,y = variables
                fig = px.density_heatmap(df_filtered, x=x, y=y, color_continuous_scale="Viridis" )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Histogram will show up automatically")



    with st.expander("Make Scatter Plot (2 or 3 D)", expanded = (len(variables)>1)):
        if len(variables) > 1:
            fig = None
            if len(variables) == 2:
                x, y = variables
                fig = px.scatter(df_filtered, x=x,y=y, color=color_col, opacity=opacity,
                                 color_continuous_scale="Viridis" , color_discrete_sequence=px.colors.qualitative.Vivid)
            if len(variables) == 3:
                x, y, z = variables
                fig = px.scatter_3d(df_filtered,x=x,y=y,z=z, color=color_col, 
                                    opacity=opacity,
                                    color_continuous_scale="Viridis", 
                                    color_discrete_sequence=px.colors.qualitative.Vivid)
            fig.update_traces(marker_size = marker_size)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Scatter plot will show up automatically")



    with st.expander("Principal Component Analysis"):
        if ss["df"] is not None:
            pp = train_xgboot.ProcessData()
            X, y = pp.preprocess(ss["df"], True)
            col1, colm, col2 = st.columns([3,1,1])
            n_components = col1.select_slider("Select No. of Principal Component", options=range(2,len(X.columns)+1 ), value = len(X.columns)) 
            process_pca = col2.button("Run PCA")

            if process_pca:
                pca = decomposition.PCA(n_components=n_components)
                pca.fit(X)
                X = pca.transform(X)
                y = np.choose(y, [1, 2, 0]).astype(float)
                fig = px.scatter_3d(x=X[:,0], y=X[:,1], z=X[:,2], color=y,
                                    opacity=opacity)
                #print(col1._html)
                fig.update_traces(marker_size = marker_size)
                st.plotly_chart(fig, use_container_width=True)
