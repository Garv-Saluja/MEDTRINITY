import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# ================================================================
# Load Models
# ================================================================
heart_model = pickle.load(open('SAVED MODELS/MEDTRINITY_Heart.sav', 'rb'))
diabetes_model = pickle.load(open('SAVED MODELS/MEDTRINITY_Diabetes.sav', 'rb'))
parkinsons_model = pickle.load(open('SAVED MODELS/MEDTRINITY_Parkinsons.sav', 'rb'))


# ================================================================
# Page Config + Custom Styling
# ================================================================
st.set_page_config(page_title="MedTrinity", layout="wide",page_icon=":heartbeat:")

st.markdown("""
    <style>
        body {background-color:#f9fafc; color:#111; font-family:'Poppins',sans-serif;}
        .main-title {text-align:center; font-size:44px; font-weight:700; color:#003566;}
        .sub-title {text-align:center; color:#495057; font-size:16px; margin-bottom:30px;}
        .stButton>button {
            background:linear-gradient(90deg,#0072ff,#00c6ff);
            color:white; border:none; border-radius:8px; padding:10px 22px;
            font-weight:600; transition:0.3s;
        }
        .stButton>button:hover {transform:scale(1.03); box-shadow:0 0 10px #00c6ff;}
        .result {font-weight:600; font-size:16px; margin-top:10px;}
        .healthy {color:#00916E;}
        .disease {color:#C1121F;}
        .graph-caption {text-align:left; font-size:12px; color:#555; margin-top:-5px;}
        .footer {text-align:center; margin-top:30px; font-size:12px; color:#6c757d;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">MedTrinity</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predicting Tomorrow, Protecting Today — Your Health, Our Technology</div>', unsafe_allow_html=True)

# ================================================================
# Sidebar Navigation
# ================================================================
with st.sidebar:
    st.header("Select a Prediction")
    selected = st.selectbox("", ['Heart Disease Prediction', 'Diabetes Prediction', "Parkinson\'s Disease Prediction"])

# ================================================================
# Helper Function for Graph Display
# ================================================================
def show_result_graph(probabilities, labels, title, caption, prediction):
    fig, ax = plt.subplots(figsize=(2, 2))
    bars = ax.bar(labels, probabilities, color=['#06d6a0', '#ef476f'], width=0.5, edgecolor='#023e8a', linewidth=1.2)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=9, color='#0072ff', pad=3)
    ax.set_xlabel("Condition", fontsize=7, color='#333')
    ax.set_ylabel("Probability", fontsize=7, color='#333')
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)

    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f"{probabilities[i]:.2f}", ha='center', va='center',
                color='white', fontsize=7, fontweight='bold')

    st.pyplot(fig, use_container_width=False)
    st.markdown(f"<p class='graph-caption'>{caption}</p>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown("<p class='result disease'>⚠️ Disease Likely</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='result healthy'>✅ Person Appears Healthy</p>", unsafe_allow_html=True)

# ================================================================
# HEART DISEASE
# ================================================================
if selected == "Heart Disease Prediction":
    st.header("Heart Disease Prediction")
    st.write("Please enter the following details:")

    age = st.slider("Age", 20, 80, 45)
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 400, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    restecg = st.selectbox("Rest ECG Results", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.radio("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.slider("ST Depression", 0.0, 6.2, 1.0)
    slope = st.selectbox("Slope of ST", ["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal", ["Normal", "Fixed Defect", "Reversible Defect"])

    if st.button("Predict Heart Disease"):
        sex = 1 if sex == "Male" else 0
        cp_enc = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp)
        fbs = 1 if fbs == "Yes" else 0
        restecg = ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
        exang = 1 if exang == "Yes" else 0
        slope = ["Upsloping", "Flat", "Downsloping"].index(slope)
        thal = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)

        data = np.array([age, sex, cp_enc, trestbps, chol, fbs, restecg,
                         thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

        pred = heart_model.predict(data)[0]
        prob = heart_model.predict_proba(data)[0]

        show_result_graph(prob, ["Healthy", "Heart Disease"],
                          "Heart Disease Prediction",
                          "Probability of being healthy vs having heart disease.", pred)

# ================================================================
# DIABETES
# ================================================================
elif selected == "Diabetes Prediction":
    st.header("Diabetes Prediction")
    st.write("Please enter the following details:")

    # ORIGINAL FEATURES
    pregnancies = st.slider("Pregnancies", 0, 17, 3)
    glucose = st.slider("Glucose", 50, 200, 100)
    blood_pressure = st.slider("Blood Pressure", 40, 120, 70)
    skin_thickness = st.slider("Skin Thickness", 0, 99, 20)
    insulin = st.slider("Insulin", 0, 846, 79)
    bmi = st.slider("BMI", 10.0, 60.0, 25.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.slider("Age", 21, 80, 35)

    # FEATURE ENGINEERING (must match training .ipynb)
    glucose_bmi_ratio = glucose / (bmi + 1e-6)
    insulin_age_ratio = insulin / (age + 1e-6)
    preg_age = pregnancies * age

    # Age groups (same binning as IPYNB)
    age_group = pd.cut(
        pd.Series([age]), 
        bins=[20,30,40,50,60,100], 
        labels=[0,1,2,3,4], 
        include_lowest=True
    ).astype(int)[0]

    # BMI groups
    bmi_group = pd.cut(
        pd.Series([bmi]),
        bins=[0,18.5,25,30,40,100],
        labels=[0,1,2,3,4],
        include_lowest=True
    ).astype(int)[0]

    if st.button("Predict Diabetes"):
        # Build dataframe EXACTLY in correct order
        input_df = pd.DataFrame([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age,
            glucose_bmi_ratio,
            insulin_age_ratio,
            preg_age,
            age_group,
            bmi_group
        ]], columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
            "Glucose_BMI_ratio",
            "Insulin_Age_ratio",
            "Preg_Age",
            "Age_group",
            "BMI_group"
        ])

        # Predictions
        pred = diabetes_model.predict(input_df)[0]
        prob = diabetes_model.predict_proba(input_df)[0]

        show_result_graph(
            prob, ["Healthy", "Diabetes"],
            "Diabetes Prediction",
            "Probability of being healthy vs diabetic.",
            pred
        )


# ================================================================
# PARKINSON’S
# ================================================================
# ================================================================
# PARKINSON’S
# ================================================================
elif selected == "Parkinson's Disease Prediction":
    st.header("Parkinson’s Disease Prediction")
    st.write("Please enter the following details:")

    fo = st.slider("MDVP:Fo(Hz)", 80, 350, 200)
    fhi = st.slider("MDVP:Fhi(Hz)", 80, 600, 300)
    flo = st.slider("MDVP:Flo(Hz)", 40, 250, 150)
    jitter = st.slider("MDVP:Jitter(%)", 0.0, 1.0, 0.5)
    jitter_abs = st.slider("MDVP:Jitter(Abs)", 0.0, 0.01, 0.005)
    shimmer = st.slider("MDVP:Shimmer", 0.0, 0.1, 0.05)
    hnr = st.slider("HNR", 10, 30, 20)
    rpde = st.slider("RPDE", 0.3, 0.7, 0.5)
    dfa = st.slider("DFA", 0.5, 1.0, 0.75)
    spread1 = st.slider("Spread1", -10.0, 10.0, 0.0)
    ppe = st.slider("PPE", 0.0, 0.5, 0.25)

    if st.button("Predict Parkinson’s"):

        # Build full 22-feature input (missing features replaced with 0)
        data = np.array([
            fo,             # 1
            fhi,            # 2
            flo,            # 3
            jitter,         # 4
            jitter_abs,     # 5
            0, 0, 0,        # 6,7,8  -> RAP, PPQ, DDP (not taken from UI)
            shimmer,        # 9
            0, 0, 0, 0,     # 10,11,12,13 -> Shimmer dB, APQ3, APQ5, APQ
            0,              # 14 -> Shimmer DDA
            0,              # 15 -> NHR
            hnr,            # 16
            rpde,           # 17
            dfa,            # 18
            spread1,        # 19
            0,              # 20 -> spread2
            0,              # 21 -> D2
            ppe             # 22
        ]).reshape(1, -1)

        pred = parkinsons_model.predict(data)[0]
        prob = parkinsons_model.predict_proba(data)[0]

        show_result_graph(
            prob, ["Healthy", "Parkinson’s"],
            "Parkinson’s Prediction",
            "Probability of being healthy vs having Parkinson’s disease.",
            pred
        )

# ================================================================
# Footer
# ================================================================
