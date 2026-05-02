import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Commercial Appeal Predictor") 

uploaded_file = st.file_uploader("Upload Excel file", type="xlsx") 

if uploaded_file is not None:
    dataset = pd.read_excel(uploaded_file)
    st.write("Data loaded:", dataset.shape) 
    
    target_col = '5. What issue does the commercial address?'

    
    if target_col in dataset.columns:
        x = dataset.drop([target_col], axis=1)
        y = dataset[target_col].astype(str).str.split('┋').str[0].str.strip()
        
        if st.button("Train model"):
            st.write("Training...")
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # One-hot encode all features including the new 'Issue' column
            x_processed = pd.get_dummies(x, drop_first=True).astype(float)
            
            x_train, x_test, y_train, y_test = train_test_split(
                x_processed, y_encoded, test_size=0.2, random_state=42
            )
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(x_train, y_train)
            
            # Save to session state
            st.session_state.x_processed = x_processed
            st.session_state.model = model
            st.session_state.le = le
            st.session_state.classes = le.classes_
            st.success(f"Model trained! Accuracy: {accuracy_score(y_test, model.predict(x_test)):.2f}")

if 'model' in st.session_state:
    st.divider()
    st.header("Brand Input: Commercial Strategy")
    
    st.subheader("Target Audience Demographics")
    new_country = st.text_input("Target Country")
    new_age = st.selectbox("Target Age", ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64"])
    new_gender = st.text_input("Gender")
    new_education = st.selectbox("Education Level", ["High school", "Apprenticeship", "Associate degree", "Bachelor degree", "Graduate or professional degree (e.g. MA or PhD)"])
    
    if st.button("Predict Best Appeal"):
        # Create a dataframe that matches the training format
        new_data = pd.DataFrame({
            '1. Where are you from?': [new_country],
            '2. How old are you?': [new_age],
            '3. How would you describe your gender identity?': [new_gender],
            '4. What is the highest level of education you have?': [new_education],
        })
        
        new_processed = pd.get_dummies(new_data).astype(float)
        new_processed = new_processed.reindex(columns=st.session_state.x_processed.columns, fill_value=0)
        
        pred = st.session_state.model.predict(new_processed)
        pred_class = st.session_state.classes[pred[0]]
        
        st.metric(label="Recommended Appeal Type", value=pred_class)
