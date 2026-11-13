"""Streamlit demo: readmission risk within 30 days.

EDUCATIONAL PROJECT ONLY: historical hospital data, not a medical
decision tool. Must not be used for clinical decisions.

Trains a LightGBM model in memory on the bundled 20,000-row stratified
sample (fast), then predicts the probability of readmission within
30 days for the entered patient profile.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from train import clean, load_data  # noqa: E402

from lightgbm import LGBMClassifier  # noqa: E402

st.set_page_config(page_title="Readmission risk — diabetes", layout="centered")
st.title("Hospital readmission within 30 days?")
# Sliders must stay left-to-right even in RTL browsers (e.g. Persian):
# without this fix, dragging the thumb right decreases the value.
st.markdown(
    """<style>
    [data-testid="stSlider"] { direction: ltr !important; }
    </style>""",
    unsafe_allow_html=True,
)
st.warning(
    "Educational portfolio demo on historical hospital data — "
    "not a medical decision tool. The model is weak by design: "
    "even the best model only reaches ROC-AUC ~0.67."
)

raw, _ = load_data()
X, y, _, _ = clean(raw)
model = LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    class_weight="balanced", n_jobs=4, verbose=-1, random_state=42,
)
model.fit(X, y)

st.subheader("Patient profile")
c1, c2 = st.columns(2)
with c1:
    age = st.slider("Age", 0, 100, 65, step=5)
    time_in_hospital = st.slider("Days in hospital", 1, 14, 4)
    num_medications = st.slider("Number of medications", 1, 40, 10)
    number_inpatient = st.slider("Prior inpatient visits (this year)", 0, 10, 0)
with c2:
    number_emergency = st.slider("Prior emergency visits", 0, 10, 0)
    number_diagnoses = st.slider("Number of diagnoses", 1, 16, 5)
    insulin = st.selectbox("Insulin used?", ["No", "Yes"])
    a1c = st.selectbox(
        "HbA1c result", ["None (not measured)", "Norm", ">7", ">8"]
    )

if st.button("Estimate readmission probability"):
    row = {col: 0 for col in X.columns}
    row["age"] = age
    row["time_in_hospital"] = time_in_hospital
    row["num_medications"] = num_medications
    row["number_inpatient"] = number_inpatient
    row["number_emergency"] = number_emergency
    row["number_diagnoses"] = number_diagnoses
    row["number_outpatient"] = 0
    row["num_lab_procedures"] = 40
    row["num_procedures"] = 1
    if insulin == "Yes":
        row["insulin"] = 1
    if a1c != "None (not measured)":
        row[f"A1Cresult_{a1c}"] = 1
    row["diabetesMed"] = 1
    row["diag_1_diabetes"] = 1
    row["diag_2_other"] = 1
    row["diag_3_other"] = 1
    row["race_Caucasian"] = 1
    row["gender_Female"] = 1
    row["discharge_disposition_id_1"] = 1
    row["admission_type_id_1"] = 1
    row["admission_source_id_7"] = 1

    import pandas as pd

    proba = model.predict_proba(pd.DataFrame([row], columns=X.columns))[0, 1]
    st.metric("Probability of readmission within 30 days", f"{proba:.1%}")
    if proba >= 0.5:
        st.error("Model flags this profile as high risk — but remember, "
                 "only ~18% of such flags are correct on this data.")
    else:
        st.success("Model flags this profile as low risk — but it still "
                   "misses ~45% of real readmissions.")
    st.caption(
        "Defaults used for fields not shown: median lab/procedure counts, "
        "diabetes as primary diagnosis, discharged to home. "
        "Full method and measured metrics: see README.md."
    )
