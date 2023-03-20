import streamlit as st
import matplotlib.pyplot as plt
from sklearn import decomposition
import numpy as np

from streamlit import session_state as ss
import plotly.express as px
import streamlit.components.v1 as components
from utils import load, setting, train_xgboot

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
    page_title="ML Training",
    page_icon="📊",
    layout="wide"
)

st.title("Data Explorer")

st.markdown(
    """
    ### Desired functions for this app:
    - Show table
    - Make 1/2/3-dimensional scatter plots of chosen quantities
    - Make 1/2-dimensional histograms of chosen quantities
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



with st.expander("Make Histogram (1 or 2 D)"):
    if len(variables) in [1,2]:
        if st.button("Make Histograms"):
            fig, ax = plt.subplots()
            if len(variables) == 1:
                x = variables[0]
                with _lock:
                    ax.hist(df[x])
                ax.set_xlabel(x)
                ax.set_ylabel("count")
            if len(variables) == 2:
                x, y = variables
                with _lock:
                    hh = ax.hist2d(df[x], df[y], cmap='viridis')
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                fig.colorbar(hh[3], ax=ax)
            st.write(fig)



with st.expander("Make Scatter Plot (2 or 3 D)"):
    if len(variables) > 1:
        if st.button("Make Scatter Plot"):
            fig, ax = plt.subplots()
            if len(variables) == 2:
                x, y = variables
                with _lock:
                    ax.plot(df[x], df[y],marker="o", linestyle='None')
                ax.set_xlabel(x)
                ax.set_ylabel(y)
            if len(variables) == 3:
                x, y, z = variables
                ax = fig.add_subplot(111, projection='3d')
                with _lock:
                    ax.plot(df[x], df[y], df[z], marker="o",linestyle='None')
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                ax.set_zlabel(z)
            st.write(fig)


with st.expander("Principal Component Analysis"):
    if ss["df"] is not None:

        pp = train_xgboot.ProcessData()
        X, y = pp.preprocess(ss["df"], True)
        col1, col2 = st.columns(2)
        n_components = col1.select_slider("Select No. of Principal Component", options=range(2,len(X.columns)+1 ), value = len(X.columns)) 
        process_pca = col2.button("Run PCA")
        if process_pca:

            pca = decomposition.PCA(n_components=n_components)
            pca.fit(X)
            X = pca.transform(X)
            y = np.choose(y, [1, 2, 0]).astype(float)
            fig = px.scatter_3d(x=X[:,0], y=X[:,1], z=X[:,2], color=y,
                                opacity=.5)
            st.plotly_chart(fig)

            #st.write(fig)

plt.show()
        



#if df is not None:
#     df = loader.extension_parser( input_file )
#     #variable selection
#     options = ["None"]+list(df.columns)
#     var_x = st.sidebar.selectbox("Select x1 from list",
#             options )

#     var_y = st.sidebar.selectbox("Select x2 from list",
#            options ) 


#     plot_type="None"
#     with st.sidebar.expander("Select histogram types (required)"):
#         plot_type = st.selectbox("Select plot type", ["None","Scatter","Histogram"])
#         if var_x != "None":
#             if df[var_x].dtype.kind in 'iufc':
#                 x_range= st.slider( "x range", value=(min(df[var_x]), max(df[var_x]) ) )
#         if var_y != "None":
#             if df[var_y].dtype.kind in 'iufc':
#                 y_range= st.slider( "y range", value=(min(df[var_y]), max(df[var_y]) ) )

#     drawPlot(df,var_x,var_y, plot_type)

# st.write(df)

    



