import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

import io
import xgboost as xgb

import asyncio
import time

from utils import load, setting, train_xgboot
from streamlit import session_state as ss


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
#title of the app
st.set_page_config(
    page_title="ML Training",
    page_icon="📊",
    #layout="wide"
)
st.title("ML Training")

#side bar
with st.sidebar:
    load.load_file()
    #subheader
    st.subheader("ML Training Settings")

    if ss["df"] is None:
        st.write("Please upload a data file.")
    else:
        with st.expander("Hyper parameter settings"):
            col1, col2 = st.columns(2)
            single_value_input = col1.checkbox("Single-value input")
            if single_value_input:
                custom_single_input = col2.checkbox("Use custom values")
                # user input a single value for each parameter
                if custom_single_input:
                    setting.set_hyperparameters_input()
                else:
                    setting.set_hyperparameters_slider()
            else:
                setting.set_hyperparameters_textlist()
        with st.expander("Preprocessing settings"):
            test_size, random_state, use_ordinal_encoder = setting.set_preprocessor_options()
# used to contain buttons that split or train data            
col1, col2 = st.sidebar.columns(2)

xgb_param = dict(learning_rate=ss["learning_rate"],
            max_depth=ss["max_depth"],
            n_estimators=ss["n_estimators"],
            colsample_bytree=ss["colsample_bytree"]
            )



#main page
        
st.markdown(
    """
    ### Training an XGB model
    Select ML Training to see training result. Select Data Viewer to inspect the uploaded and splitted data.
    """
)

    

tab1, tab2, tab3 = st.tabs(["ML Training", "Data Viewer", "Parameter Viewer"])

with tab1:
    # Show training status and final plots. 
    st.markdown(
        """
        ## Instruction
        This app trains an XGB model using a standard data format. 
        User could specify a range of hyperparameters in the sidebar:
        - Uncheck Single-value input to specify a range of values for each parameter.
        - Check Single-value input to select only 1 value for each parameter.

        User could also specify data splitting options.
        Press Split Data to preprocess and split the data. 
        The Training button becomes clickable once the data has been splitted.
    """
    )


    st.write("Training status")
   
    data_split = False
    if col1.button("Prep and Split", disabled=(ss["df"] is None)):
        ss["splitted"] = train_xgboot.preprocess_and_split_data(test_size,random_state,use_ordinal_encoder)
    
    trained = col2.button("Train XGB Model", disabled=(not ss["splitted"]) )
    if trained:
        # The typical training time for the test DataFrame is ~44 s 
        # on my PC. I imagine an user might experience much longer 
        # wait time because the data is larger, or more parameters
        # are used.
        # TODO: Add a timer
        #   Example here: https://medium.com/@soumyas567/make-a-timer-with-streamlit-15ef3e340b7e and
        #                 https://discuss.streamlit.io/t/how-to-make-a-timer/22675
        # These timers will countdown from pre-defined starting point, then exit upon t=0. Not what I want
        # It seems I need to use async functions
        #   Example here: https://discuss.streamlit.io/t/issue-with-asyncio-run-in-streamlit/7745/2
        # I'll experiment with it another day
        # def write_time(ts):
        #     st.write("%02d:%05.2f"%divmod(ts,60))

        # async def stopwatch(ts, run=True):
        #     while run:
        #         write_time(ts)
        #         time.sleep(0.01)
        #         ts+=0.01

        with st.spinner("Training ..."):
            model = train_xgboot.XGBmodel(xgb_param)
            model.fit(ss["X_train"],ss["y_train"])
            ss["model"] = model
            ss["trained"] = True

    # Make ROC and CM plots once model is trained
    # Use state to persist the bolean option across app refresh
    if ss["trained"] and ss["model"]:
        st.success("Training complete!")
        # draw plots
        ac1, ac2 = st.columns(2)
        timestamp = str(time.time()).replace(".","T")
        with _lock:
            fig1, ax1 = plt.subplots()
            ss["model"].show_test_result_roc(ss["X_test"], ss["y_test"],ax=ax1)
            ss["fig1"] = fig1
            fig2, ax2 = plt.subplots()
            ss["model"].show_test_result_confusion(ss["X_test"], ss["y_test"],ax=ax2)
            ss["fig2"] = fig2
            ac1.pyplot(ss["fig1"])
            ac2.pyplot(ss["fig2"])

            ac1, ac2 = st.columns(2)
            img1 = io.BytesIO()
            fig1.savefig(img1, format='pdf')
            img2 = io.BytesIO()
            fig2.savefig(img2, format='pdf')
            

            ac1.download_button("Download ROC", data=img1, mime="application/pdf", file_name="XGB-ROC-%s.pdf"%timestamp)
            ac2.download_button("Download CM", data=img2, mime="application/pdf", file_name="XGB-ConfusionMatrix-%s.pdf"%timestamp)

        




with tab2:
    with st.expander("Display input data"):
        if ss["df"] is not None:
            st.dataframe(ss["df"])
    with st.expander("Display preprocessed input data"):
        if ss["formated_df"] is not None:
            st.dataframe(ss["formated_df"])
    with st.expander("Display split data"):
        st.write("Training sample:")
        tc1,tc2 = st.columns(2)
        tc1.write(ss["X_train"]) 
        tc2.write(ss["y_train"]) 

        st.write("Test sample:")
        tc1,tc2 = st.columns(2)
        tc1.write(ss["X_test"])
        tc2.write(ss["y_test"])


with tab3:
    with st.expander("Hyperparameter settings"):
        st.write(xgb_param)
    with st.expander("Model parameters"):
        if ss["model"] is not None:
            st.write(ss["model"].clf.get_params())
        else:
            st.write("Model parameters will appear here after training.")

    if ss["trained"] is True:
        # well this will save a local copy of tmp.json on the server, which is not 
        # always desirable. 
        # TODO: somehow save the json in memory?
        ss["model"].clf.save_model("tmp.json")
        with open("tmp.json") as f:
            st.download_button("Download Model (json)",data = f, file_name = f"model-{timestamp}.json", )
    else:
        st.download_button("Download Model (json)",data = "s", disabled=True )



