import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import *
import base64
import io
import requests


# ------------------------------------------------------------------------------------------
# 定義

# 定義 Download Function
def download_link(object_to_download, download_filename, download_link_text):

    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


# 定義 Background
page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1567427361940-521d3e67e193?ixlib=rb-1.2.1&ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&auto=format&fit=crop&w=708&q=80");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# https://images.unsplash.com/photo-1493606278519-11aa9f86e40a?ixlib=rb-1.2.1&ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&auto=format&fit=crop&w=1050&q=80


st.markdown(
    f"""
# <style>
    .main{{
        font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans",sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji";
    }}
    .main .block-container .markdown-text-container{{
        background-color: #FFFFFF;
    }}
    .fullScreenFrame{{
        background-color: #FFFFFF;
    .streamlit-button small-button primary-button {{
        background-color: #FFFFFF;
    }}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------------------
# Sidebar Section


# Default File
url = "https://raw.githubusercontent.com/Chiehcode/Drug_Classification_Web_App/main/Data/Default.csv"
url_data = requests.get(url).content
sample_data = pd.read_csv(io.StringIO(url_data.decode('utf-8')))

# User Input
st.sidebar.header('Input Features')


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader(
    "Upload your Input Features in CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

else:
    st.sidebar.markdown(
        '<div style="font-size: small">or select the input features below:</div>',
        unsafe_allow_html=True)

    def user_input_features():
        age = st.sidebar.slider(
            'Age', 0, 100, 30)
        spr = st.sidebar.slider(
            'Sodium to Potassium Ratio', 0.0, 20.0, 10.0)
        sex = st.sidebar.selectbox(
            'Sex', ('F', 'M'))
        blood_pressure = st.sidebar.selectbox(
            'Blood Pressure', ('HIGH', 'LOW', 'NORMAL'))
        cholesterol = st.sidebar.selectbox(
            'Cholesterol', ('HIGH', 'NORMAL'))

        data = {'Sex': sex,
                'Blood Pressure': blood_pressure,
                'Cholesterol': cholesterol,
                'Age': age,
                'Sodium to Potassium Ratio': spr}

        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()


# ------------------------------------------------------------------------------------------
# Main Section

# App Title
st.markdown(
    '<h1>Drug Classification App</h1>',
    unsafe_allow_html=True)

# st.markdown(
#     '<h1 style="color:#22577a;">Drug Classification App</h1>',
#     unsafe_allow_html=True)


st.write("""
請在左方欄位以「上傳檔案」或「手動輸入」的方式鍵入特徵值，以預測 Drug Type。

Data obtained from [Drug Classification Dataset](https://www.kaggle.com/prathamtripathi/drug-classification)
""")

# Displays the user input features
st.subheader('Inputted Features:')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write(input_df)

# Apply Download Function for Sample Data
if st.button('下載範例格式'):
    tmp_download_link = download_link(
        sample_data, 'Sample.csv', 'Click here to download Sample Data')
    st.markdown(tmp_download_link, unsafe_allow_html=True)


# load the model
final_model = load_model('Final Decision Tree Model')


# Apply model to make predictions
prediction = final_model.predict(input_df)
prediction_proba = final_model.predict_proba(input_df)


# -------------------------------------------
# Prediction
st.subheader('Prediction:')

drug = np.array(['DrugY', 'DrugA', 'DrugB', 'DrugC', 'DrugX'])

df = pd.DataFrame({
    'Prediction': drug[prediction]
})

st.write(df)

# Apply Download Function for Predition Data
if st.button('下載預測結果'):
    tmp_download_link = download_link(
        df, 'Prediction.csv', 'Click here to download Predition Data')
    st.markdown(tmp_download_link, unsafe_allow_html=True)


# -------------------------------------------
# Prediction Probability

st.subheader('Prediction Probability:')

df2 = pd.DataFrame(prediction_proba, columns=drug)

st.write(df2)


# Apply Download Function for Prediction Probability
if st.button('下載預測概率'):
    tmp_download_link = download_link(
        df2, 'Prediction_Probability.csv', 'Click here to download Prediction Probability Data')
    st.markdown(tmp_download_link, unsafe_allow_html=True)
