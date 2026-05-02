import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="AdInsight AI", page_icon="🧠", layout="wide")

# 2. SIDEBAR BRANDING
with st.sidebar:
    st.title("🧠 AdInsight AI")
    st.markdown("---")
    st.info("Upload your survey data to train the engine, then enter your target demographic to predict the most effective commercial issue.")
    st.markdown("Developed for **Brand Strategy Optimization**")

# 3. INITIALIZE MEMORY
if 'trained' not in st.session_state:
    st.session_state.trained = False

st.title("Commercial Issue Predictor")

# 4. DATA LOADING SECTION
uploaded_file = st.file_uploader("Upload your survey results (XLSX)", type="xlsx")

if uploaded_file is not None:
    dataset = pd.read_excel(uploaded_file)
    
    # --- DATA MAPPING ---
    # Connects specific survey ads to broader societal issues
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
        return "Other Social Issue"

    dataset['Issue'] = dataset['7. Which advertisement appeals to you the most?'].apply(map_ad_to_issue)
    
    feature_cols = [
        '1. Where are you from?', 
        '2. How old are you?', 
        '3. How would you describe your gender identity?', 
        '4. What is the highest level of education you have?'
    ]

    # Use an expander for the technical training part
    with st.expander("🛠️ AI Engine Training"):
        if st.button("🚀 Train Model"):
            with st.status("Analyzing demographics...", expanded=True) as status:
                X = dataset[feature_cols]
                y = dataset['Issue']
                
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                X_processed = pd.get_dummies(X, drop_first=True).astype(float)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_processed, y_encoded)
                
                st.session_state.model = model
                st.session_state.le = le
                st.session_state.X_columns = X_processed.columns
                st.session_state.trained = True
                status.update(label="AI Model Optimized!", state="complete", expanded=False)
            st.success("The engine is ready for prediction.")

# 5. PREDICTION INTERFACE
if st.session_state.trained:
    st.divider()
    st.header("🎯 Predict Optimal Commercial Strategy")
    
    # Input Dashboard using Columns
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            new_country = st.text_input("Target Country", value="Germany")
            new_age = st.selectbox("Target Age Group", ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64"])
        with c2:
            new_gender = st.selectbox("Target Gender", ["Female", "Male", "Non-binary", "Other"])
            new_edu = st.selectbox("Target Education", ["High school", "Bachelor degree", "Graduate degree"])

        predict_btn = st.button("Generate Strategy Analysis", type="primary", use_container_width=True)

    if predict_btn:
        # Prepare data
        new_data = pd.DataFrame({
            '1. Where are you from?': [new_country],
            '2. How old are you?': [new_age],
            '3. How would you describe your gender identity?': [new_gender],
            '4. What is the highest level of education you have?': [new_edu]
        })
        
        new_processed = pd.get_dummies(new_data).astype(float)
        new_processed = new_processed.reindex(columns=st.session_state.X_columns, fill_value=0)
        
        # Get prediction and probabilities
        prediction = st.session_state.model.predict(new_processed)
        probs = st.session_state.model.predict_proba(new_processed)[0]
        result_issue = st.session_state.le.inverse_transform(prediction)[0]
        
        # Display Results
        st.markdown("### Analysis Results")
        
        # Top metric showing the winner
        st.metric(label="Primary Recommended Issue", value=result_issue)
        
        # Visualization of Market Resonance
        prob_df = pd.DataFrame({
            'Issue': st.session_state.le.classes_,
            'Appeal Probability': probs
        }).sort_values('Appeal Probability', ascending=True)

        st.markdown("#### Market Resonance Probability")
        st.bar_chart(prob_df.set_index('Issue'), horizontal=True)
        
        st.toast("Analysis complete!", icon="✅")
else:
    if uploaded_file:
        st.warning("Please expand the 'AI Engine Training' section and click 'Train Model' to begin.")
