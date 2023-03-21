# streamlit-app
## Installation
> pip install -r requirements.txt

Please note if you expect users to upload LibreOffice files, you will also need to

> pip install odfpy
## Start app
On your terminal, do 

> streamlit run Info.py

## Input data description
Users of this app could upload a csv/xslx data file for simple exploratory data analysis and training using XGBClassifier. 

Presently the data file must contain the following columns:

Category columns:
```
gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod
```
Numerical columns:
```
tenure, MonthlyCharges, TotalCharges, Churn, and customerID
```
The app does a column name check, and will return failure if the format is not exactly correct. 
A future version may allow users to dynamically select data columns but that feature seems to be geared more towards advanced audience. 

### Upload data
Just drag a file and drop it into the file box. 

# Subpages
The app consists of three pages -- `Info`, `Data Explorer`, `ML Training`
## Info
> Info page!

## Data Explorer
Users could make 2/3-dimensional scatter plots and 1/2-dimensional histograms of selected features in the dataset. 
### Sidebar: Preprocessing
Clicking on the preprocessing button will create a copy of the data with the categorical data encoded with numbers. 
### Main page: Plotting
Users could select which data file -- the original or the preprocessed data -- for plotting, they also have the choice to select only numeric columns.
The options to make scatter or histograms will dynamically appear in the main page as users select the desired features. 
### Main page: PCA
Users could select the number of features to keep for PCA analysis. An interactive 3D figure of the leading features will appear once PCA has been processed. 


## ML Training
Users could view data, train model, download ROC and Confusion Matrix plots, view model parameters and download trained model configuration. 
### Sidebar: Preprocessing, Split, and Train
Users could select from a varietie of hyper-parameter settings for the training process, and select splitting fraction and random state from the expandable menu.
### Main page Tab: ML Training
Once a model is trained, ROC and CM plots will appear here, and users could download them using the associated download button. 
### Main page: Data Viewer
Users could inspect the original, the preprocessed, and the splitted data tables.
### Main page: Parameter Viewer
Users could inspect some of the ML model parameters


# Running in Docker
1) Build the docker image with 
> docker build . --network=host -t streamlit-app

2) Run the docker container:
>docker run -it --rm -p 4546:4545 streamlit-app

The command will create and run a docker container built from `streamlit-app`, and expose the docker internal port 4545 to external port 4546.

Users can then open their browser and view the app at `http://localhost:4546`.
