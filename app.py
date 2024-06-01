import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data_encoders():
    model = pickle.load(open("model.pkl", "rb"))
    data = pd.read_csv("Thyroid_Diff.csv")

    label_encoders = {
        'Gender': LabelEncoder(),
        'Smoking': LabelEncoder(),
        'Hx Smoking': LabelEncoder(),
        'Thyroid Function': LabelEncoder(),
        'Physical Examination': LabelEncoder(),
        'Adenopathy': LabelEncoder(),
        'Pathology': LabelEncoder(),
        'Focality': LabelEncoder(),
        'Risk': LabelEncoder(),
        'T': LabelEncoder(),
        'N': LabelEncoder(),
        'M': LabelEncoder(),
        'Stage': LabelEncoder(),
        'Response': LabelEncoder()
    }

    for column in data.drop(["Recurred", "Hx Radiothreapy"], axis=1).select_dtypes("object").columns.tolist():
        label_encoders[column].fit(data[column])
  
    return label_encoders, model

def get_input_values(label_encoders):
    st.header("🧬 General Measurments")
    
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", options=label_encoders['Gender'].classes_)
    smoking = st.selectbox("Smoking", options=label_encoders['Smoking'].classes_)
    hx_smoking = st.selectbox("Smoking History", options=label_encoders['Hx Smoking'].classes_)
    thyroid_function = st.selectbox("Thyroid Function", options=label_encoders['Thyroid Function'].classes_)
    physical_examination = st.selectbox("Physical Examination", options=label_encoders['Physical Examination'].classes_)
    adenopathy = st.selectbox("Adenopathy", options=label_encoders['Adenopathy'].classes_)
    pathology = st.selectbox("Pathology", options=label_encoders['Pathology'].classes_)
    focality = st.selectbox("Focality", options=label_encoders['Focality'].classes_)
    risk = st.selectbox("Risk", options=label_encoders['Risk'].classes_)
    T = st.selectbox("Tumor Сlassification", options=label_encoders['T'].classes_)
    N = st.selectbox("Lymph  Nodal Сlassification", options=label_encoders['N'].classes_)
    M = st.selectbox("Metastasis Сlassification", options=label_encoders['M'].classes_)
    stage = st.selectbox("Stage", options=label_encoders['Stage'].classes_)
    response = st.selectbox("Response", options=label_encoders['Response'].classes_)

    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Smoking': [smoking],
        'Hx Smoking': [hx_smoking],
        'Thyroid Function': [thyroid_function],
        'Physical Examination': [physical_examination],
        'Adenopathy': [adenopathy],
        'Pathology': [pathology],
        'Focality': [focality],
        'Risk': [risk],
        'T': [T],
        'N': [N],
        'M': [M],
        'Stage': [stage],
        'Response': [response]
    })

    for column in label_encoders:
        input_data[column] = label_encoders[column].transform(input_data[column])

    age_scaler = StandardScaler()
    input_data['Age'] = age_scaler.fit_transform(input_data[['Age']])

    return input_data

def add_predictions(model, input_values):
    prediction = model.predict(input_values)
    probability = model.predict_proba(input_values)

    if prediction[0] == 0:
        st.write("<span class='pred low'>Low Chances of cancer recurrence</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='pred high'>High Chances of cancer recurrence</span>", unsafe_allow_html=True)

    st.write(f"Chances of Recurrence: {(probability[0][1] * 100).round(3)}%")


def main():
    st.set_page_config(page_title="Cancer Recurrence Predictor",
                    page_icon=":hospital:",
                    layout="centered")
    
    with open("styles.css") as styles:
        st.markdown("<style>{}</style>".format(styles.read()), unsafe_allow_html=True)
    
    st.title(":hospital: Thyroid Cancer Recurrence Predictor")
    st.write("""The majority of thyroid cancers are called well differentiated thyroid cancers, 
                meaning the cells retain important features of normal thyroid cells when they become
                malignant. Well differentiated thyroid cancers can be further categorized as papillary
                thyroid cancer (the most common) and follicular thyroid cancer.""")
    st.markdown("<hr>", unsafe_allow_html=True) 
    st.write("Following Web Applicaiton is written using Streamlit Python library and deployed machine learnning model to accurately predict recurrence of thyroid cancer enabling doctors to assess whether cancer returns after a period of remission")

    encoder, model = load_data_encoders()
    input_values = get_input_values(encoder)
    btn = st.button("Calculate Probability")

    if btn:
        add_predictions(model, input_values)


if __name__ == "__main__":
    main()