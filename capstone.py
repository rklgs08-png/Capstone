import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Commercial Issue Predictor") 

uploaded_file = st.file_uploader("Upload Excel file", type="xlsx") 

if uploaded_file is not None:
    dataset = pd.read_excel(uploaded_file)
    st.write("Data loaded:", dataset.shape) 
  
    
    target_col = '5. What issue does the commercial address?'
    
    if target_col not in dataset.columns:
        st.error(f"Could not find column: '{target_col}'")
        st.write("Available columns:", list(dataset.columns))
    else:
        st.success(f"Found {target_col}! You can now train.")
    
    if target_col in dataset.columns:
        # Features (X) includes demographic info + the Appeal type
        x = dataset.drop([target_col], axis=1)
        y = dataset[target_col].astype(str).str.split('┋').str[0].str.strip()
        
        if st.button("Train model"):
            st.write("Training...")
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Convert all text features (including Appeal) into numbers
            x_processed = pd.get_dummies(x, drop_first=True).astype(float)
            
            x_train, x_test, y_train, y_test = train_test_split(
                x_processed, y_encoded, test_size=0.2, random_state=42
            )
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(x_train, y_train)
            
            # Save variables to use them later in the prediction section
            st.session_state.x_processed = x_processed
            st.session_state.model = model
            st.session_state.le = le
            st.session_state.classes = le.classes_
            
            # Find the unique appeals available in the data for the dropdown later
            # Assuming '7. Which advertisement appeals to you the most?' is the appeal column
            appeal_col = '7. Which advertisement appeals to you the most?'
            st.session_state.appeal_options = dataset[appeal_col].unique().tolist()
            
            st.success(f"Model trained! Accuracy: {accuracy_score(y_test, model.predict(x_test)):.2f}")

if 'model' in st.session_state:
    st.divider()
    st.header("Predict the Best Issue to Address")
    
   
    st.subheader("Commercial Strategy")
   
    new_appeal = st.selectbox("Which advertisement appeal will you use?", st.session_state.appeal_options)
    
    st.subheader("Target Audience")
    new_country = st.text_input("Target Country")
    new_age = st.selectbox("Target Age", ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64"])
    new_gender = st.text_input("Gender")
    new_education = st.selectbox("Education Level", ["High school", "Apprenticeship", "Associate degree", "Bachelor degree", "Graduate or professional degree (e.g. MA or PhD)"])
    
    if st.button("Predict Optimal Issue"):
        # Create dataframe matching the training features
        # IMPORTANT: Column names must match the Excel exactly
        new_data = pd.DataFrame({
            '1. Where are you from?': [new_country],
            '2. How old are you?': [new_age],
            '3. How would you describe your gender identity?': [new_gender],
            '4. What is the highest level of education you have?': [new_education],
            '7. Which advertisement appeals to you the most?': [new_appeal]
        })
        
        # Process the input just like the training data
        new_processed = pd.get_dummies(new_data).astype(float)
        new_processed = new_processed.reindex(columns=st.session_state.x_processed.columns, fill_value=0)
        
        # Predict
        pred = st.session_state.model.predict(new_processed)
        pred_class = st.session_state.classes[pred[0]]
        
        st.success("Analysis Complete")
        st.metric(label="Recommended Issue to Address", value=pred_class)
