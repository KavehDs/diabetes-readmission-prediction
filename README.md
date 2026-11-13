# Diabetes Hospital Readmission Prediction

🚀 **Live demo:** [Try the app](https://njsq2g7jm5yc5qzdv39x4s.streamlit.app/)

Predict whether a diabetic patient will be readmitted to hospital within
30 days of discharge — from 101,766 real encounters in 130 US hospitals
(1999–2008) — and be honest about how hard this problem is.

> ⚠️ **This is an educational portfolio project on historical hospital
> data, not a medical decision tool. It must not be used for clinical
> decisions.** The data are ten-plus-year-old administrative records,
> not a live clinical system, and the models below are far from good
> enough to decide anyone's care.

## The honest headline: this problem has a low ceiling

Only **11.2%** of encounters end in readmission within 30 days, and the
signal in administrative data is weak. Prior work on this dataset
reports ROC-AUC roughly **0.57–0.69**, and our models land in exactly
that range. A model that predicts "nobody is readmitted" scores 88.8%
accuracy and finds zero cases — accuracy is the wrong metric here.

## Results (test set, n = 19,899; 2,236 readmitted <30 days)

Stratified 80/20 train/test split (`random_state=42`) on 99,492 cleaned
rows. Positive class = readmitted within 30 days.

| Model | Accuracy | ROC-AUC | Precision (<30d) | Recall (<30d) | F1 (<30d) | Brier |
|---|---|---|---|---|---|---|
| **XGBoost** | 0.6676 | **0.6683** | 0.1805 | **0.5537** | **0.2722** | 0.2140 |
| Logistic Regression | 0.6703 | 0.6635 | 0.1783 | 0.5367 | 0.2677 | 0.2269 |
| LightGBM | 0.6732 | 0.6626 | 0.1794 | 0.5345 | 0.2686 | 0.2100 |
| Majority-class baseline | 0.8877 | 0.5000 | — | 0.0000 | 0.0000 | — |

Reading the numbers: the best model finds about 55% of the patients who
will be readmitted, but only ~18% of its "will be readmitted" flags are
right. That is the reality of this dataset, not a bug in the code.

## What drives readmission (SHAP, XGBoost, 300-row test sample)

1. `number_inpatient` — prior inpatient visits (0.309)
2. `discharge_disposition_id_11` — **discharged as "Expired"** (0.179)
3. `discharge_disposition_id_1` — discharged to home (0.160)
4. `number_diagnoses` (0.072)
5. `num_medications` (0.067)

A warning about driver #2: patients recorded as "Expired" cannot be
readmitted by definition, so that feature partly encodes the outcome
rather than predicting it. Many published pipelines drop expired and
hospice discharges; we kept them and flag this openly instead of
quietly deleting the inconvenient rows.

## Fairness check (XGBoost, recall for the <30-day class)

| Group | Recall | ROC-AUC |
|---|---|---|
| Female | 0.561 | 0.672 |
| Male | 0.545 | 0.664 |
| Caucasian | 0.564 | 0.673 |
| AfricanAmerican | 0.529 | 0.648 |
| Hispanic | 0.475 | 0.650 |
| Asian | 0.727 | 0.860 |
| Age 18–44 | 0.50–0.74 | 0.67–0.81 |
| Age 65+ | 0.49–0.58 | 0.61–0.67 |

Recall is lowest for Hispanic patients (0.48) and for the "Other" race
group (0.31), and for the youngest age bracket [10–20) (0.29, small
group). The Asian group looks best but is small, so that number is
noisy. Full per-group numbers (with group sizes) are in `metrics.json`.

## Method

- **Cleaning** — the point of this project. Dropped `weight` (96.9%
  missing), `payer_code` (39.6% missing) and `medical_specialty`
  (49.1% missing); dropped 3 rows with gender "Unknown/Invalid" and
  rows with missing race (2.2%). The 24 medication features (No /
  Steady / Up / Down) became binary "used". `diag_1/2/3` ICD-9 codes
  were grouped into diabetes / circulatory / respiratory / digestive /
  injury / genitourinary / neoplasms / other / missing. Age brackets
  became midpoints. Identifiers (`encounter_id`, `patient_nbr`) were
  dropped.
- **A data quirk, stated plainly:** 101,766 encounters come from only
  71,518 patients, so the same patient can appear several times.
  Encounters are treated as independent rows; a patient-level split
  would be stricter and is listed as future work.
- **Models:** Logistic Regression (scaled, class-weighted), XGBoost
  (300 trees) and LightGBM (300 trees), all with class-imbalance
  handling. No model wins on every metric.

## Files

```
├── README.md
├── app.py                  # Streamlit demo
├── requirements.txt
├── metrics.json            # all measured numbers
├── data/
│   └── readmission_sample.csv   # 20,000-row stratified sample (bundled)
├── plots/
│   ├── confusion_matrix.svg
│   ├── roc_curves.svg
│   └── shap_top_drivers.svg
└── src/
    └── train.py
```

The full 19 MB `diabetic_data.csv` is not in this repo; `src/train.py`
downloads it from UCI on first run and falls back to the bundled sample
with a printed warning.

## Reproduce

```bash
pip install -r requirements.txt
python src/train.py        # ~25 s on a laptop CPU
streamlit run app.py
```

## Data source & licence

Diabetes 130-US Hospitals for Years 1999–2008, Strack et al. 2014,
UCI Machine Learning Repository (#296), **CC BY 4.0**:
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
