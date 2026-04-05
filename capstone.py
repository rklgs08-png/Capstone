import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

st.title=("Commercial Appeal Predictor")

uploaded_file=st.file_uploader("395k0As85_Text_When Brands Take Sides Public Reactions to Political Advertising_43_43.xlsx", type="xlsl")
if uploaded_file is not None:
    dataset = pd.read_excel(uploaded_file)
    st.write=("Data loaded:", dataset.shape)
    x=dataset.drop(['7. Which advertisement appeals to you the most?'], axis=1)
    y=dataset['7. Which advertisement appeals to you the most?']
    if st.button("Train model"):
        st.write("Training...")
        le=LabelEncoder()
        y_encoded=le.fit_transform(y)
        x_processed= pd.get_dummies(x, drop_first=True).astype(float)
        x_train, x_test, y_train, y_test = train_test_split(x_processed, y_encoded, test_size=0.2, random_state=42)
        model=RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        accuracy=accuracy_score(y_test,y_pred)
        st.success(f"Model trained! Accuracy: {accuracy:.2f}")
        st.write ("Ad classes:", list (le.classes_))

        st.session_state.model=model
        st.session_state.le=le
        st.session_state.classes=le.classes_
    if 'model' in st.session_state:
        st.header("Make Prediction")
        new_country=st.text_input ("Country")
        new_age=st.selectbox("Age", ["Under 18", "18-24", "25-34", "35-44", "45-54","55-64"])
        new_gender=st.text_input("Gender")
        if st.button("Predict"):
            new_data=pd.DataFrame({
                '1. Where are you from?': [new_country], 
                '2. How old are you?': [new_age], 
                '3. How would you describe your gender identity?': [new_gender]})
                new_processed=pd.get_dummies(new_data, drop_first=True).astype(float)
                new_processed=new_processed.reindex(columns=x_processed.columns, fill_value=0)

                pred=st.session_state.model.predict(new_processed)
                pred_class=st.session_state.classes[pred[0]]
                pred_proba=st.session_state.model.predict_proba(new_processed)[0]
                st.write (f "**Predict ad appeal:** {pred_class}")
                st.write ("**Probabilities:**")
                for i, prob in enumerate(pred_proba):
                    st.write(f" {st.session_state.classes[i]}: {prob:.2f}")
                      
    
