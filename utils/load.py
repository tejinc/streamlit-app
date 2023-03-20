import pandas as pd
import streamlit as st
from streamlit import session_state as ss
from streamlit.runtime.uploaded_file_manager import UploadedFile as UploadedFile 

excel_types = ["xls", "xlsx", "xlsm", "xlsb", "odf", "ods", "odt"]
csv_types = ["csv"]

def get_types():
    return csv_types+excel_types

def extension_parser( uploaded_file: UploadedFile ) -> pd.DataFrame:
    print(uploaded_file.name)
    ext = uploaded_file.name.split(".")[-1].lower()

    # return 
    if ext == 'csv':
        return pd.read_csv(uploaded_file)
    elif ext in ["xls", "xlsx", "xlsm", "xlsb", "odf", "ods", "odt"]:
        return pd.read_excel(uploaded_file)

    return None

def load_file(uploader_label = "Please upload csv or xslx files"):
    ss["input_file"] = st.file_uploader( uploader_label, type = get_types() )
    if ss["input_file"] is not None:
        ss["df"] = extension_parser( ss["input_file"] )
    if ss["df"] is not None and ss["input_file"] is not None:
        st.success(f"%s has been uploaded!"%(ss["input_file"].name) )

def unload_file():
    del ss["input_file"]
    del ss["df"]
    ss["input_file"] = None
    ss["df"] = None

def unload_file_button(label="Delete File"):
    st.button(label, on_click=unload_file)