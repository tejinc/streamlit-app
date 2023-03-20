import re
import streamlit as st
from streamlit import session_state as ss

ml_parameters = ['learning_rate','max_depth', 'n_estimators', 'colsample_bytree']
file_settings = ["input_file", "df", "formated_df", "X_train", "X_test", "y_train", "y_test", "model"]
file_loading_states = ["file_loaded", "trained", "splitted", "plot_scatter", "plot_hist"]

def initialize_setting():
    for s in file_settings:
        if s not in ss:
            ss[s] = None

    for s in file_loading_states:
        if s not in ss:
            ss[s] = False

    # a) in the window of learning_rate, user can input [0.01, 0.016, 0.027, 0.046, 0.077, 0.129, 0.215, 0.359],
    # b) in the window of max_depth, user can input [5, 10, 15],
    # c) in the window of n_estimators, user can input [ 50,  70,  90, 110, 130, 150],
    # d) in the window of colsample_bytree, user can input [0.4, 0.6, 0.8]
    for s in ml_parameters:
        if s not in ss:
            ss[s] = []



def set_hyperparameters_slider():
    ss["learning_rate"] =       [st.select_slider(label="learning_rate",
                                           options=[0.01, 0.016, 0.027, 0.046, 0.077, 0.129, 0.215, 0.359],
                                           format_func=float
                                           )]
    ss["max_depth"] =           [st.select_slider(label="max_depth",
                                           options=[5,10,15],
                                           format_func=int
                                           )]
    ss["n_estimators"] =        [st.select_slider(label="n_estimators",
                                           options=[ 50,  70,  90, 110, 130, 150],
                                           format_func=int
                                           )]
    ss["colsample_bytree"] =    [st.select_slider(label="colsample_bytree",
                                           options=[0.4, 0.6, 0.8],
                                           format_func=float
                                           )]
def set_hyperparameters_input():
    ss["learning_rate"] =   [st.number_input( label="learning_rate"   )] #, value = values[0])
    ss["max_depth"] =       [st.number_input( label="max_depth"       )] #, value = values[1])
    ss["n_estimators"] =    [st.number_input( label="n_estimators"    )] #, value = values[2])
    ss["colsample_bytree"] =[st.number_input( label="colsample_bytree")] #, value = values[3])



def is_float(element: any) -> bool:
    #If you expect None to be passed:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def convert_string_to_numbers(input_text: str, delimiter=",", num_type=float)->list:
    split_text = input_text.split(delimiter)
    if num_type == float:
        ret = list(map(float,[ t for t in split_text if is_float(t) ]))
    else:
        ret = list(map(int,[ t for t in split_text if is_float(t) ]))
    #print("test",ret)
    return ret
def set_hyperparameters_textlist():
    text_help="input a list of number separated by comma (,)"
    value1="0.01, 0.016, 0.027, 0.046, 0.077, 0.129, 0.215, 0.359"
    value2="5, 10, 15"
    value3="50, 70, 90, 110, 130, 150"
    value4="0.4, 0.6, 0.8"
    ss["learning_rate"] =       convert_string_to_numbers( st.text_input(help=text_help, value=value1, placeholder=value1, label="learning_rate"    ), delimiter=",", num_type=float )
    ss["max_depth"] =           convert_string_to_numbers( st.text_input(help=text_help, value=value2, placeholder=value2, label="max_depth"        ), delimiter=",", num_type=int   )
    ss["n_estimators"] =        convert_string_to_numbers( st.text_input(help=text_help, value=value3, placeholder=value3, label="n_estimators"     ), delimiter=",", num_type=int   )
    ss["colsample_bytree"] =    convert_string_to_numbers( st.text_input(help=text_help, value=value4, placeholder=value4, label="colsample_bytree" ), delimiter=",", num_type=float )

def set_preprocessor_options():
    #feature_columns=None,columns_to_drop=None, target_column = None, test_size=0.3, random_state=22, use_ordinal_encoder=True
    test_size = st.number_input("Test Size", value=0.3, max_value=1., min_value=0.)
    random_state = st.number_input("Random state", value=42,step=1)

    #this option has been set to true because the data requires.
    use_ordinal_encoder = st.checkbox("Use Ordinal Encoder", value = True, disabled=True)
    return test_size, random_state, use_ordinal_encoder