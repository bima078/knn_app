# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
import matplotlib
matplotlib.use('Agg')

# Step 1: Read the Data
try:
    data = pd.read_excel('data\Book1.xlsx')
    print("Data loaded successfully")
except Exception as e:
    print("Error loading data:", e)
    raise e

# Display the data
st.dataframe(data)

data.info()

# Ubah semua tipe data menjadi float
data = data.astype(float)

# Cek data yang hilang
data.isnull().sum()

# Menghapus atribut keyword difficulty
columnDrop = ['Keyword Difficulty']
data = data.drop(columnDrop, axis=1)

# Cek data apakah berhasil di drop atau tidak
duplicateData = data.duplicated()
data[duplicateData]

# Menentukan target klasifikasi
data['Tingkat Trending'].value_counts()

x = data.drop("Tingkat Trending", axis=1).values
y = data.iloc[:,-1]

# Visualisasi data target
data['Tingkat Trending'].value_counts().plot(kind='bar', figsize=(10,6), color=['green', 'blue'])
plt.title("Count of the target")
plt.xticks(rotation=1)
plt.show()

# Oversampling dengan SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
xSmote_resampled, ySmote_resampled = smote.fit_resample(x, y)

# Visualisasi data sebelum dan sesudah oversampling
plt.figure(figsize=(12, 4))
newDF1 = pd.DataFrame(data=y)

plt.subplot(1, 2, 1)
newDF1.value_counts().plot(kind='bar', figsize=(10, 6), color=['green', 'blue'])
plt.title("Target before Oversampling with SMOTE")
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
newDF2 = pd.DataFrame(data=ySmote_resampled)

newDF2.value_counts().plot(kind='bar', figsize=(10, 6), color=['green', 'blue'])
plt.title("Target after Oversampling with SMOTE")
plt.xticks(rotation=0)

# Standarisasi data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
xSmote_resampled_normal = scaler.fit_transform(xSmote_resampled)

dfCek1 = pd.DataFrame(xSmote_resampled_normal)
dfCek1.describe()

# Membagi data train dan test
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(xSmote_resampled, ySmote_resampled, test_size=0.4, random_state=42, stratify=ySmote_resampled)

xTrain_normal, xTest_normal, yTrain_normal, yTest_normal = train_test_split(xSmote_resampled_normal, ySmote_resampled, test_size=0.4, random_state=42, stratify=ySmote_resampled)

# Fungsi evaluasi
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix, classification_report

def evaluation(yTest, yPred):
    accTest = accuracy_score(yTest, yPred)
    rclTest = recall_score(yTest, yPred, average='weighted')
    f1Test = f1_score(yTest, yPred, average='weighted')
    psTest = precision_score(yTest, yPred, average='weighted')

    metric_dict = {'accuracy Score': round(accTest, 3),
                   'recall Score': round(rclTest, 3),
                   'f1 Score': round(f1Test, 3),
                   'Precision Score': round(psTest, 3)}

    return print(metric_dict)

# Pemodelan KNN
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(xTrain, yTrain)

yPred_knn = knn_model.predict(xTest)

# Evaluasi KNN Model
print("K-Nearest Neighbors (KNN) Model: ")
accuracy_knn_smote = round(accuracy_score(yTest, yPred_knn), 3)

print("Accuracy: ", accuracy_knn_smote)
print("Classification Report: ")
print(classification_report(yTest, yPred_knn))

# Visualisasi Confusion Matrix
confMatrix = confusion_matrix(yTest, yPred_knn)

plt.figure(figsize=(8, 6))
sns.heatmap(confMatrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("True")
plt.ylabel("Predict")
plt.show()

modelComp1 = pd.DataFrame({'Model': ['K-Nearest Neighbour'], 'Accuracy': [accuracy_knn_smote * 100, ]})

fig, ax = plt.subplots()
bars = plt.bar(modelComp1['Model'], modelComp1['Accuracy'], color=['green'])
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Oversample')
plt.xticks(rotation=45, ha='right')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

# Simpan model KNN ke file menggunakan pickle
with open('model\knn_model.pkl', 'wb') as file:
    pickle.dump(knn_model, file)

# Muat model KNN dari file menggunakan pickle
with open('model\knn_model.pkl', 'rb') as file:
    loaded_knn_model = pickle.load(file)

# STREAMLIT
st.set_page_config(page_title="Klasifikasi Keyword Pencarian Produk PMB Toys Menggunakan KNN")

st.title("Klasifikasi Keyword Pencarian Produk PMB Toys Menggunakan KNN")
st.write(f"**_Model's Accuracy_** :  :green[**{accuracy_knn_smote * 100}**]% (:red[_Do not copy outright_])")
st.write("")

tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])

with tab1:
    st.sidebar.header("**User Input** Sidebar")

    volume = st.sidebar.number_input(label=":violet[**Volume**]", min_value=data['Volume'].min(), max_value=data['Volume'].max())
    st.sidebar.write(f":orange[Min] value: :orange[**{data['Volume'].min()}**], :red[Max] value: :red[**{data['Volume'].max()}**]")
    st.sidebar.write("")

    trend_apr = st.sidebar.number_input(label=":violet[**Trend April'23**]", min_value=data["Trend April'23"].min(), max_value=data["Trend April'23"].max())
    trend_may = st.sidebar.number_input(label=":violet[**Trend Mei'23**]", min_value=data["Trend Mei'23"].min(), max_value=data["Trend Mei'23"].max())
    trend_jun = st.sidebar.number_input(label=":violet[**Trend Juni'23**]", min_value=data["Trend Juni'23"].min(), max_value=data["Trend Juni'23"].max())
    trend_jul = st.sidebar.number_input(label=":violet[**Trend Juli'23**]", min_value=data["Trend Juli'23"].min(), max_value=data["Trend Juli'23"].max())
    trend_aug = st.sidebar.number_input(label=":violet[**Trend Agu'23**]", min_value=data["Trend Agu'23"].min(), max_value=data["Trend Agu'23"].max())
    trend_sep = st.sidebar.number_input(label=":violet[**Trend Sep'23**]", min_value=data["Trend Sep'23"].min(), max_value=data["Trend Sep'23"].max())
    trend_oct = st.sidebar.number_input(label=":violet[**Trend Okt'23**]", min_value=data["Trend Okt'23"].min(), max_value=data["Trend Okt'23"].max())
    trend_nov = st.sidebar.number_input(label=":violet[**Trend Nov'23**]", min_value=data["Trend Nov'23"].min(), max_value=data["Trend Nov'23"].max())
    trend_dec = st.sidebar.number_input(label=":violet[**Trend Dec'23**]", min_value=data["Trend Dec'23"].min(), max_value=data["Trend Dec'23"].max())
    trend_jan = st.sidebar.number_input(label=":violet[**Trend Jan'24**]", min_value=data["Trend Jan'24"].min(), max_value=data["Trend Jan'24"].max())
    trend_feb = st.sidebar.number_input(label=":violet[**Trend Feb'24**]", min_value=data["Trend Feb'24"].min(), max_value=data["Trend Feb'24"].max())
    trend_mar = st.sidebar.number_input(label=":violet[**Trend Mar'24**]", min_value=data["Trend Mar'24"].min(), max_value=data["Trend Mar'24"].max())

    inputArray = [volume, trend_apr, trend_may, trend_jun, trend_jul, trend_aug, trend_sep, trend_oct, trend_nov, trend_dec, trend_jan, trend_feb, trend_mar]

    st.subheader("User Input Value")
    st.write(f"""
    **Volume** : {volume} |
    **Trend April'23** : {trend_apr} |
    **Trend Mei'23** : {trend_may} |
    **Trend Juni'23** : {trend_jun} |
    **Trend Juli'23** : {trend_jul} |
    **Trend Agu'23** : {trend_aug} |
    **Trend Sep'23** : {trend_sep} |
    **Trend Okt'23** : {trend_oct} |
    **Trend Nov'23** : {trend_nov} |
    **Trend Dec'23** : {trend_dec} |
    **Trend Jan'24** : {trend_jan} |
    **Trend Feb'24** : {trend_feb} |
    **Trend Mar'24** : {trend_mar} |
    """)

    st.write("")
    if st.button("Predict Class"):
        classValue = loaded_knn_model.predict([inputArray])
        st.subheader("Predicted Class")
        st.write(f"**Tingkat Trending** : :violet[**{classValue[0]}**]")
        st.write("")

with tab2:
    st.subheader("Multiple Prediction from CSV")

    uploaded_file = st.file_uploader("Choose a CSV file for prediction")

    if uploaded_file is not None:
        dataMultiPredict = pd.read_csv(uploaded_file)

        st.write(f"Data from file : {uploaded_file.name}")
        st.write(dataMultiPredict)

        dataMultiPredict['predicted_class'] = loaded_knn_model.predict(dataMultiPredict)

        st.write("")
        st.write("Prediction Results")
        st.write(dataMultiPredict)
        st.write("")
        csv = dataMultiPredict.to_csv(index=False)
        st.download_button(label="Download the prediction results", data=csv, file_name='prediction_results.csv')
