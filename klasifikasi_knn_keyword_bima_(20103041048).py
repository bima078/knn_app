# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# Load the data
try:
    data = pd.read_excel('data/Book1.xlsx')
    data = data.astype(float)
    data.drop(['Keyword Difficulty'], axis=1, inplace=True)
except Exception as e:
    st.write("Error loading data:", e)
    raise e

# Oversample the data using SMOTE
x = data.drop("Tingkat Trending", axis=1).values
y = data.iloc[:, -1].values
smote = SMOTE(random_state=42)
x_smote_resampled, y_smote_resampled = smote.fit_resample(x, y)

# Normalize the data
scaler = MinMaxScaler()
x_smote_resampled_normal = scaler.fit_transform(x_smote_resampled)

# Split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_smote_resampled, y_smote_resampled, test_size=0.4, random_state=42, stratify=y_smote_resampled)
x_train_normal, x_test_normal, y_train_normal, y_test_normal = train_test_split(x_smote_resampled_normal, y_smote_resampled, test_size=0.4, random_state=42, stratify=y_smote_resampled)

# Load the KNN model from file using pickle
with open('model/knn_model.pkl', 'rb') as file:
    loaded_knn_model = pickle.load(file)

# Evaluate the model
def evaluation(y_test, y_pred):
    acc_test = accuracy_score(y_test, y_pred)
    rcl_test = recall_score(y_test, y_pred, average='weighted')
    f1_test = f1_score(y_test, y_pred, average='weighted')
    ps_test = precision_score(y_test, y_pred, average='weighted')

    metric_dict = {
        'accuracy Score': round(acc_test, 3),
        'recall Score': round(rcl_test, 3),
        'f1 Score': round(f1_test, 3),
        'Precision Score': round(ps_test, 3)
    }

    return metric_dict

# STREAMLIT
st.set_page_config(page_title="Klasifikasi Keyword Pencarian Produk PMB Toys Menggunakan KNN")

st.title("Klasifikasi Keyword Pencarian Produk PMB Toys Menggunakan KNN")
st.write(f"**Model's Accuracy**: {accuracy_score(y_test, loaded_knn_model.predict(x_test)) * 100}%")

tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])

with tab1:
    st.header("Single Prediction")
    st.write("Masukkan nilai untuk melakukan prediksi tunggal.")

    volume = st.number_input(label="Volume", min_value=float(data['Volume'].min()), max_value=float(data['Volume'].max()))
    trend_apr = st.number_input(label="Trend April'23", min_value=float(data["Trend April'23"].min()), max_value=float(data["Trend April'23"].max()))
    trend_may = st.number_input(label="Trend Mei'23", min_value=float(data["Trend Mei'23"].min()), max_value=float(data["Trend Mei'23"].max()))
    trend_jun = st.number_input(label="Trend Juni'23", min_value=float(data["Trend Juni'23"].min()), max_value=float(data["Trend Juni'23"].max()))
    trend_jul = st.number_input(label="Trend Juli'23", min_value=float(data["Trend Juli'23"].min()), max_value=float(data["Trend Juli'23"].max()))
    trend_aug = st.number_input(label="Trend Agu'23", min_value=float(data["Trend Agu'23"].min()), max_value=float(data["Trend Agu'23"].max()))
    trend_sep = st.number_input(label="Trend Sep'23", min_value=float(data["Trend Sep'23"].min()), max_value=float(data["Trend Sep'23"].max()))
    trend_oct = st.number_input(label="Trend Okt'23", min_value=float(data["Trend Okt'23"].min()), max_value=float(data["Trend Okt'23"].max()))
    trend_nov = st.number_input(label="Trend Nov'23", min_value=float(data["Trend Nov'23"].min()), max_value=float(data["Trend Nov'23"].max()))
    trend_dec = st.number_input(label="Trend Dec'23", min_value=float(data["Trend Dec'23"].min()), max_value=float(data["Trend Dec'23"].max()))
    trend_jan = st.number_input(label="Trend Jan'24", min_value=float(data["Trend Jan'24"].min()), max_value=float(data["Trend Jan'24"].max()))
    trend_feb = st.number_input(label="Trend Feb'24", min_value=float(data["Trend Feb'24"].min()), max_value=float(data["Trend Feb'24"].max()))
    trend_mar = st.number_input(label="Trend Mar'24", min_value=float(data["Trend Mar'24"].min()), max_value=float(data["Trend Mar'24"].max()))

    input_array = [volume, trend_apr, trend_may, trend_jun, trend_jul, trend_aug, trend_sep, trend_oct, trend_nov, trend_dec, trend_jan, trend_feb, trend_mar]

    if st.button("Predict"):
        class_value = loaded_knn_model.predict([input_array])
        st.subheader("Predicted Class")
        st.write(f"Tingkat Trending: {class_value[0]}")

with tab2:
    st.header("Multiple Predictions from CSV")
    st.write("Upload file CSV untuk melakukan prediksi multiple.")

    uploaded_file = st.file_uploader("Choose a CSV file")

    if uploaded_file is not None:
        data_multi_predict = pd.read_csv(uploaded_file)
        predictions = loaded_knn_model.predict(data_multi_predict)

        st.subheader("Prediction Results")
        st.write(data_multi_predict.assign(predicted_class=predictions))

        csv = data_multi_predict.to_csv(index=False)
        st.download_button(label="Download Prediction Results", data=csv, file_name='prediction_results.csv')
