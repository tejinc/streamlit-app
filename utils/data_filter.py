import streamlit as st
import matplotlib.pyplot as plt
from sklearn import decomposition
import numpy as np

from streamlit import session_state as ss
import plotly.express as px
import streamlit.components.v1 as components
from utils import load, setting, train_xgboot


def data_filter(df):
    sel_options_num = dict()
    sel_options_cat = dict()
    df_filtered = df

    # select columns to filter
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = list(set(df.columns)-set(num_cols))

    # obtain options for numerical columns
    num_cols_to_filter = st.multiselect("Filter numeric columns:", options = sorted(num_cols))
    sel_options_num.clear()
    for f in num_cols_to_filter:
        vmin, vmax = min(df[f]), max(df[f])
        sel_options_num[f] = st.slider(f,min_value=vmin,max_value=vmax, value=(vmin,vmax)  )
    st.write(sel_options_num)

    # obtain options for categorical columns
    cat_cols_to_filter = st.multiselect("Filter categorical columns", options = sorted(cat_cols))
    sel_options_cat.clear()
    for f in cat_cols_to_filter:
        opts = list(set(df[f]))
        sel_options_cat[f] = st.multiselect(f,options=opts,default=opts )
    st.write(sel_options_cat)

    # apply limits to num cols.
    for f, vals in sel_options_num.items():
        vmin,vmax = vals
        df_filtered = df_filtered.loc[(df[f] >= vmin) & (df[f]<=vmax)]

    # apply or conditions to cat cols --> assuming selected values must be true
    for f, cats in sel_options_cat.items():
        l = [ df_filtered[f] == cat for cat in cats]
        df_filtered = df_filtered.loc[np.logical_or.reduce(l)]
    return df_filtered