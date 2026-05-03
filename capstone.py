import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="AdInsight | Green Edition", 
    page_icon="🌿", 
    layout="wide"
)

st.markdown("""
    <style>
    /* Main background color */
    .stApp {
        background-color: #f7fdf9;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #e8f5e9;
    }
    
    /* Customizing buttons */
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
    }
    
    div.stButton > button:hover {
        background-color: #45a049;
        color: white;
        border: 2px solid #45a049;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("🌿 AdInsight")
    st.markdown("---")
    st.markdown("### **Navigation**")
    st.info("Upload your data to generate demographic-based predictions.")
    st.markdown("---")
    st.caption("v2.0 | Green Strategy Mode")


if 'trained' not in st.session_state:
    st.session_state.trained = False

st.title("Commercial Issue Predictor")
st.markdown("##### :green[Analyze and predict social resonance for brand campaigns]")


uploaded_file = st.file_uploader("🎯Step 1: Upload Survey Results (XLSX)", type="xlsx")

if uploaded_file is not None:
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
        return "Other Social Issue"

    dataset['Issue'] = dataset['7. Which advertisement appeals to you the most?'].apply(map_ad_to_issue)
    
    feature_cols = [
        '1. Where are you from?', 
        '2. How old are you?', 
        '3. How would you describe your gender identity?', 
        '4. What is the highest level of education you have?'
    ]

    with st.expander("🛠️ Configuration"):
        if st.button("🚀 Train Engine"):
            with st.status("Processing survey data...", expanded=True) as status:
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
                status.update(label="Training Complete!", state="complete", expanded=False)
            st.toast("AI model is now live!", icon="🌿")


if st.session_state.trained:
    st.divider()
    st.header("🎯 Step 2: Predict Strategy")
    
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            new_country = st.text_input("Target Country", value="Germany")
            new_age = st.selectbox("Target Age Group", ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64"])
        with c2:
            new_gender = st.selectbox("Target Gender", ["Female", "Male", "Non-binary", "Other"])
            new_edu = st.selectbox("Target Education", ["High school", "Bachelor degree", "Graduate degree"])

        predict_btn = st.button("Run Market Analysis", use_container_width=True)

    if predict_btn:
        # Data Prep
        new_data = pd.DataFrame({
            '1. Where are you from?': [new_country],
            '2. How old are you?': [new_age],
            '3. How would you describe your gender identity?': [new_gender],
            '4. What is the highest level of education you have?': [new_edu]
        })
        
        new_processed = pd.get_dummies(new_data).astype(float)
        new_processed = new_processed.reindex(columns=st.session_state.X_columns, fill_value=0)
        
        # Prediction & Probabilities
        prediction = st.session_state.model.predict(new_processed)
        probs = st.session_state.model.predict_proba(new_processed)[0]
        result_issue = st.session_state.le.inverse_transform(prediction)[0]
        
        # Results Display
        st.markdown("### Analysis Results")
        
        col_metric, col_chart = st.columns([1, 2])
        
        with col_metric:
            st.metric(label="Primary Recommended Issue", value=result_issue)
            st.write("---")
            st.write(f"The model suggests focusing on **{result_issue}** to maximize appeal for this specific group.")
        
        with col_chart:
            # Create a colorful Plotly bar chart
            prob_df = pd.DataFrame({
                'Issue': st.session_state.le.classes_,
                'Probability': probs
            }).sort_values('Probability', ascending=True)

            fig = px.bar(
                prob_df, 
                x='Probability', 
                y='Issue', 
                orientation='h',
                title="Market Resonance Probability",
                color='Probability',
                color_continuous_scale='Greens' # Light green to dark green gradient
            )
            fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        st.toast("Insights generated!", icon="📊")
else:
    if uploaded_file:
        st.warning("Model ready for training. Please click 'Train Engine' above.")
