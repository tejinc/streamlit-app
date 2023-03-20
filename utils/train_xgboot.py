from streamlit import session_state as ss
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from scikitplot import metrics 
import inspect, re
import time
import timeit
import warnings
warnings.filterwarnings('ignore')


######### class of XGBoost model with functions including:  
######### model fitting, Hyperparameter tuning, target variable prediction, and output performance plots

@st.cache_resource
class XGBmodel(object):
    
    def __init__(self,param):
        self.param=param
        
    
    def fit(self, X, y, random_state=22):
        clf = xgb.XGBClassifier(random_state=random_state)
        clf_search=RandomizedSearchCV(estimator=clf,n_iter=50,param_distributions=self.param, scoring='accuracy',cv=10,n_jobs=4)
        clf_search.fit(X,y)
        self.clf = clf_search.best_estimator_
        return self
        
    def predict_proba(self, X):
        res_proba = self.clf.predict_proba(X)
        return res_proba
    
    def predict(self, X):
        res = self.clf.predict(X)
        return res
    

    def show_test_result_roc(self,X,y,ax=None):
        result_proba=self.predict_proba(X)
        metrics.plot_roc(y,result_proba, ax = ax)

    def show_test_result_confusion(self,X,y,ax=None):
        predictions = cross_val_predict(self.clf, X, y)
        metrics.plot_confusion_matrix(y, predictions, normalize=True, ax = ax)
       
        
    def precision_recall_f1_visual(self,X,y):
        print(classification_report(y,self.predict(X),digits=4))



class ProcessData(object):
    def __init__(self) -> None:
        self.category_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                'PaymentMethod']
        self.numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.drop_feature_cols = ['Churn', 'customerID']
        self.label_column = 'Churn'

    @st.cache_data
    def check_dataframe(_self, tcc):
        if set(tcc.columns) == set(_self.category_cols+_self.numeric_cols+_self.drop_feature_cols):
            return True
        else:
            return False

    @st.cache_data
    def preprocess(_self, tcc, use_ordinal_encoder=True):
        if _self.check_dataframe(tcc):
            try:
                tcc['TotalCharges']= tcc['TotalCharges'].apply(lambda x: x if x!= ' ' else np.nan).astype(float)
                tcc['MonthlyCharges'] = tcc['MonthlyCharges'].astype(float)
                tcc['TotalCharges'] = tcc['TotalCharges'].fillna(0)
                tcc['Churn'].replace(to_replace='Yes', value=1, inplace=True)
                tcc['Churn'].replace(to_replace='No',  value=0, inplace=True)
                features = tcc.drop(columns=_self.drop_feature_cols).copy()
                labels = tcc['Churn'].copy()

                if use_ordinal_encoder:
                    ord_enc = OrdinalEncoder()
                    ord_enc.fit(tcc[_self.category_cols])
                    features_OE= pd.DataFrame(ord_enc.transform(features[_self.category_cols]), columns=_self.category_cols)
                    features_OE.index = features.index
                    features_OE = pd.concat([features_OE, features[_self.numeric_cols]], axis=1)
                    features = features_OE
            except: 
                st.error("Preprocessing failed! Check column names")
                return None, None
            else:
                st.success("Preprocessing succeeded!")
                return features, labels
        else:
            st.error("Column names do not match.")
        
    @st.cache_data
    def split_data(_self, features, labels, test_size=0.3,random_state=22):
        if features is None or labels is None:
            raise Exception("Please preprocess data first")
            return False
        try:
            X_train, X_test,y_train,y_test = train_test_split(features,labels, test_size=test_size, random_state=random_state)
            # if use_ordinal_encoder:
            #     numeric_cols = X_train._get_numeric_data().columns
            #     category_cols =  list(set(X_train.columns) - set(numeric_cols))
            #     ord_enc = OrdinalEncoder()
            #     ord_enc.fit(X_train[_self.category_cols])
            #     X_train_OE = pd.DataFrame(ord_enc.transform(X_train[_self.category_cols]), columns=_self.category_cols)
            #     X_train_OE.index = X_train.index
            #     X_train_OE = pd.concat([X_train_OE, X_train[_self.numeric_cols]], axis=1)
 
            #     X_test_OE = pd.DataFrame(ord_enc.transform(X_test[_self.category_cols]), columns=_self.category_cols)
            #     X_test_OE.index = X_test.index
            #     X_test_OE = pd.concat([X_test_OE, X_test[_self.numeric_cols]], axis=1)
 
            #     X_train = X_train_OE
            #     X_test = X_test_OE
            st.success("Data split succeeded!")
            return X_train, y_train, X_test, y_test
            # self.X_train = X_train
            # self.X_test = X_test
            # self.y_train = y_train
            # self.y_test = y_test
            # st.write(X_train)
        except:
            st.error("Data split failed!")
            return None, None, None, None

def format_data():
    if ss["df"] is not None:
        preprocessor = ProcessData()
        preprocessor.check_dataframe(ss["df"])
        features, labels = preprocessor.preprocess(ss["df"], True)
        ss["formated_df"] =pd.concat([features,labels], axis=1)
        return True
    return False

def preprocess_and_split_data(test_size, random_state, use_ordinal_encoder):
    if ss["df"] is not None:
        preprocessor = ProcessData()
        preprocessor.check_dataframe(ss["df"])
        features, labels = preprocessor.preprocess(ss["df"], use_ordinal_encoder)
        ss["formated_df"] = pd.concat([features,labels], axis=1)
        ss["X_train"], ss["y_train"], ss["X_test"], ss["y_test"]= preprocessor.split_data(features, labels, test_size,random_state)
        return True
    return False