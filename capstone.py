import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.title("Commercial Issue Predictor")

uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")

if uploaded_file is not None:
    # Read the file
    dataset = pd.read_excel(uploaded_file)

    ad_to_issue = {
        "Dream Crazy": "Racial Injustice",
        "Ford's First Icon": "Gender Equality",
        "Love Conquers All": "LGBTQ+ Rights",
        "Don't Buy This Jacket": "Environmentalism"
    }

    def map_ad_to_issue(ad_text):
        for key, value in ad_to_issue.items():
            if key in str(ad_text):
                return value
        return "Other/General Social"

    dataset['Issue'] = dataset['7. Which advertisement appeals to you the most?'].apply(map_ad_to_issue)
    
    feature_cols = [
        '1. Where are you from?', 
        '2. How old are you?', 
        '3. How would you describe your gender identity?', 
        '4. What is the highest level of education you have?'
    ]
    
    target_col = 'Issue'

    if st.button("Train Model"):
        st.write("Training model to predict the best issue...")

        X = dataset[feature_cols]
        y = dataset[target_col]
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        X_processed = pd.get_dummies(X, drop_first=True).astype(float)
        
      
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_processed, y_encoded)
        
       
        st.session_state.model = model
        st.session_state.le = le
        st.session_state.X_columns = X_processed.columns
        st.session_state.trained = True
        st.success("Model trained! Ready to predict.")

if st.session_state.get('trained'):
    st.divider()
    st.header("Brand Strategy: Find Your Focus")
    
    st.subheader("Target Audience Details")
    new_country = st.text_input("Country")
    new_age = st.selectbox("Age Group", ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64"])
    new_gender = st.selectbox("Gender Identity", ["Female", "Male", "Non-binary", "Other"])
    new_edu = st.selectbox("Education", ["High school", "Bachelor degree", "Graduate degree"])

    if st.button("Predict Issue"):
        new_data = pd.DataFrame({
            '1. Where are you from?': [new_country],
            '2. How old are you?': [new_age],
            '3. How would you describe your gender identity?': [new_gender],
            '4. What is the highest level of education you have?': [new_edu]
        })
        
      
        new_processed = pd.get_dummies(new_data).astype(float)
        new_processed = new_processed.reindex(columns=st.session_state.X_columns, fill_value=0)
        
        prediction = st.session_state.model.predict(new_processed)
        result_issue = st.session_state.le.inverse_transform(prediction)[0]
        
        st.metric(label="Recommended Social Issue to Address", value=result_issue)
        st.write("Based on your target audience's demographics, this issue is most likely to generate high appeal.")
