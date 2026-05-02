import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Commercial Appeal Predictor")

TARGET = "7. Which advertisement appeals to you the most?"
FEATURES = [
    "1. Where are you from?",
    "2. How old are you?",
    "3. How would you describe your gender identity?",
    "4. What is the highest level of education you have?",
]

uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    df = df[FEATURES + [TARGET]].copy()
    df[TARGET] = df[TARGET].astype(str).str.split("┋").str[0].str.strip()

    X = df[FEATURES]
    y = df[TARGET]

    st.write("Rows:", df.shape[0])

    if st.button("Train model"):
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        cat_cols = FEATURES
        preprocessor = ColumnTransformer([
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ])

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))

        st.success(f"Model trained! Accuracy: {acc:.2f}")
        st.write("Classes:", list(le.classes_))

        st.session_state.model = model
        st.session_state.le = le

if "model" in st.session_state:
    st.header("Make Prediction")

    country = st.text_input("Country")
    age = st.selectbox("Age", ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64"])
    gender = st.selectbox("Gender", ["Female", "Male", "Non-binary","Other"])
    education = st.selectbox(
        "Education",
        ["High school", "Apprenticeship", "Associate degree", "Bachelor degree", "Graduate or professional degree (e.g. MA or PhD)"]
    )
    important = st.selectbox("Important for brands?", ["yes", "no"])

    if st.button("Predict"):
        new_data = pd.DataFrame([{
            "1. Where are you from?": country,
            "2. How old are you?": age,
            "3. How would you describe your gender identity?": gender,
            "4. What is the highest level of education you have?": education,
            "6. Do you think it is important for brands to address political/societal issues in commercials?": important,
        }])

        proba = st.session_state.model.predict_proba(new_data)[0]
        pred_idx = np.argmax(proba)
        pred_class = st.session_state.le.inverse_transform([pred_idx])[0]

        st.write(f"**Predicted ad appeal:** {pred_class}")
        for cls, p in zip(st.session_state.le.classes_, proba):
            st.write(f"{cls}: {p:.2f}")
