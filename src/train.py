"""Train and evaluate readmission models on the Diabetes 130-US Hospitals
dataset (1999-2008, UCI #296, Strack et al. 2014), CC BY 4.0.

Run:  python src/train.py
It prints the measured test-set metrics and writes metrics.json plus the
plots in plots/.

Data: 101,766 encounters from 130 US hospitals, 47 features, target
`readmitted` (<30 days / >30 days / never). Binary task here: readmitted
within 30 days (11.2%) vs not.

The full CSV (data/diabetic_data.csv, ~19 MB) is gitignored and
downloaded at runtime if missing; the bundled data/readmission_sample.csv
(20,000 rows, stratified) is used as a fallback with a printed warning.

Cleaning (the point of this project — the data are messy):
- Drop identifiers encounter_id, patient_nbr. Note: 101,766 encounters
  come from only 71,518 patients, so the same patient can appear several
  times; encounters are treated as independent rows (a stated limitation).
- Drop `weight` (96.9% missing), `payer_code` (39.6% missing) and
  `medical_specialty` (49.1% missing) — too empty to impute honestly.
- Drop 3 rows with gender 'Unknown/Invalid'; drop rows with missing race
  (2.2%) — small enough to drop without biasing the sample.
- 24 medication features (No / Steady / Up / Down) -> binary "used"
  (any of Steady/Up/Down = 1). `change` and `diabetesMed` -> binary.
- diag_1/2/3 ICD-9 codes grouped into: diabetes, circulatory,
  respiratory, digestive, injury, genitourinary, neoplasms, other,
  missing. (Codes starting with V/E or '?' -> other / missing.)
- max_glu_serum and A1Cresult one-hot encoded (most rows are 'None',
  i.e. the test was not performed).
- age brackets [0-10) ... [90-100) -> bracket midpoint (numeric).
- race, gender, admission_type_id, admission_source_id and
  discharge_disposition_id one-hot encoded; rare discharge codes grouped
  into 'other'.

EDUCATIONAL PROJECT ONLY: historical hospital data, not a medical
decision tool. It must not be used for clinical decisions.
"""

import json
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FULL = DATA / "diabetic_data.csv"
SAMPLE = DATA / "readmission_sample.csv"
UCI_ZIP = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
RANDOM_STATE = 42

MEDS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


def download_full():
    print("Full data not found — downloading UCI zip (may take a minute)...")
    zip_path = DATA / "diabetic_data.zip"
    urllib.request.urlretrieve(UCI_ZIP, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extract("diabetic_data.csv", DATA)
    zip_path.unlink()
    print("Downloaded", FULL)


def load_data():
    if FULL.exists():
        print("Using full file:", FULL)
        return pd.read_csv(FULL, low_memory=False), "full"
    if SAMPLE.exists():
        print("WARNING: full file missing — falling back to bundled 20,000-row sample")
        return pd.read_csv(SAMPLE, low_memory=False), "sample"
    download_full()
    return pd.read_csv(FULL, low_memory=False), "full"


def icd_group(code):
    """Group an ICD-9 code into a broad category."""
    if pd.isna(code) or code == "?":
        return "missing"
    s = str(code).strip()
    if s.startswith(("V", "E")):
        return "other"
    try:
        n = float(s)
    except ValueError:
        return "other"
    if n == 250 or 250 <= n < 251:
        return "diabetes"
    if 390 <= n <= 459 or n == 785:
        return "circulatory"
    if 460 <= n <= 519 or n == 786:
        return "respiratory"
    if 520 <= n <= 579 or n == 787:
        return "digestive"
    if 800 <= n <= 999:
        return "injury"
    if 580 <= n <= 629 or n == 788:
        return "genitourinary"
    if 140 <= n <= 239:
        return "neoplasms"
    return "other"


def clean(df):
    df = df.copy()
    n0 = len(df)
    df = df.drop(columns=["encounter_id", "patient_nbr"])
    # Drop the three near-empty / very sparse columns
    df = df.drop(columns=["weight", "payer_code", "medical_specialty"])
    df = df[df["gender"] != "Unknown/Invalid"]
    df = df[df["race"] != "?"]
    kept_index = df.index
    # Target: readmitted within 30 days
    y = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])
    # Age bracket -> midpoint
    age_mid = {f"[{a}-{a+10})": a + 5 for a in range(0, 100, 10)}
    df["age"] = df["age"].map(age_mid)
    # Medications -> binary "used"
    for m in MEDS:
        df[m] = (df[m] != "No").astype(int)
    df["change"] = (df["change"] == "Ch").astype(int)
    df["diabetesMed"] = (df["diabetesMed"] == "Yes").astype(int)
    # ICD-9 diagnosis groups
    for c in ["diag_1", "diag_2", "diag_3"]:
        df[c] = df[c].apply(icd_group)
    # One-hot the remaining categoricals (missing test results stay as 'None')
    cat_cols = [
        "race", "gender", "diag_1", "diag_2", "diag_3",
        "max_glu_serum", "A1Cresult",
        "admission_type_id", "discharge_disposition_id",
        "admission_source_id",
    ]
    # Group rare discharge dispositions into 'other'
    counts = df["discharge_disposition_id"].value_counts()
    rare = set(counts[counts < 500].index)
    df["discharge_disposition_id"] = df["discharge_disposition_id"].apply(
        lambda v: "other" if v in rare else v
    )
    df = pd.get_dummies(df, columns=cat_cols, dtype=int)
    return df, y, n0, kept_index


def evaluate(name, y_true, y_pred, y_proba):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_positive": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_positive": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_positive": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "brier": round(float(brier_score_loss(y_true, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    raw, source_used = load_data()
    print(f"Loaded {len(raw):,} rows, {raw.shape[1]} columns (source: {source_used})")
    class_counts = raw["readmitted"].value_counts().to_dict()
    print("Target counts:", class_counts)
    X, y, n0, kept_index = clean(raw)
    print(f"After cleaning: {len(X):,} rows kept of {n0:,}, {X.shape[1]} features")

    # Bundled stratified sample (20,000 rows) for the repo
    sample_idx = train_test_split(
        kept_index, test_size=20000, random_state=RANDOM_STATE,
        stratify=y,
    )[1]
    raw.loc[sample_idx].to_csv(SAMPLE, index=False)
    print("Bundled sample written:", SAMPLE, f"({len(sample_idx):,} rows)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train {len(X_train):,} / test {len(X_test):,} "
          f"(positive rate train {y_train.mean():.4f}, test {y_test.mean():.4f})")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
            n_jobs=4, random_state=RANDOM_STATE,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", n_jobs=4, verbose=-1,
            random_state=RANDOM_STATE,
        ),
    }

    results, probas = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = evaluate(name, y_test, pred, proba)
        probas[name] = proba
        r = results[name]
        print(f"{name}: acc {r['accuracy']} | ROC-AUC {r['roc_auc']} | "
              f"P {r['precision_positive']} R {r['recall_positive']} "
              f"F1 {r['f1_positive']} | Brier {r['brier']}")

    # Baseline: always predict "not readmitted"
    base = evaluate("Majority baseline", y_test,
                    np.zeros_like(y_test), np.zeros(len(y_test)))
    results["Majority-class baseline"] = base
    print(f"Majority baseline: acc {base['accuracy']} | ROC-AUC 0.5")

    best_name = max((k for k in results if k != "Majority-class baseline"),
                    key=lambda k: results[k]["roc_auc"])
    print("Best by ROC-AUC:", best_name)

    # ---- Fairness check (best model) ----
    best_model = models[best_name]
    best_proba = probas[best_name]
    raw_test = raw.loc[y_test.index]
    fairness = {}
    for col, label in [("age", "age"), ("race", "race"), ("gender", "gender")]:
        fairness[label] = {}
        for grp, idx in raw_test.groupby(col).groups.items():
            mask = y_test.index.isin(idx)
            if mask.sum() < 50 or y_test[mask].sum() < 5:
                continue
            p = best_proba[mask]
            fairness[label][str(grp)] = {
                "n": int(mask.sum()),
                "positive_rate": round(float(y_test[mask].mean()), 4),
                "recall_positive": round(float(recall_score(
                    y_test[mask], (p >= 0.5).astype(int), zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_test[mask], p)), 4)
                if y_test[mask].nunique() > 1 else None,
            }
    print("Fairness (", best_name, "):")
    for k, v in fairness.items():
        print(" ", k, {g: (d["recall_positive"], d["roc_auc"]) for g, d in v.items()})

    # ---- SHAP (best tree model) ----
    shap_model = best_name if best_name in ("XGBoost", "LightGBM") else "LightGBM"
    shap_explainer_model = models[shap_model]
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_test), size=min(300, len(X_test)), replace=False)
    explainer = shap.TreeExplainer(shap_explainer_model)
    sv = explainer.shap_values(X_test.iloc[idx])
    if isinstance(sv, list):
        sv = sv[1]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:10]
    shap_top = [
        {"feature": str(X_test.columns[i]),
         "mean_abs_shap": round(float(mean_abs[i]), 4)}
        for i in order
    ]
    print("SHAP top 10:", shap_top)

    metrics = {
        "dataset": {
            "name": "Diabetes 130-US Hospitals, 1999-2008 (UCI #296, Strack et al. 2014)",
            "canonical_source": "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
            "mirror_used": UCI_ZIP,
            "licence": "CC BY 4.0",
            "rows_used": int(len(raw)),
            "columns": int(raw.shape[1]),
            "encounters": int(len(raw)),
            "unique_patients": int(raw["patient_nbr"].nunique()),
            "target": "readmitted within 30 days (binary: <30 vs rest)",
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "positive_rate": round(float(y.mean()), 4),
            "split": "stratified 80/20, random_state=42",
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "source_used": source_used,
            "cleaning": {
                "dropped_columns": ["encounter_id", "patient_nbr", "weight (96.9% missing)",
                                    "payer_code (39.6% missing)",
                                    "medical_specialty (49.1% missing)"],
                "dropped_rows": "3 rows gender Unknown/Invalid; rows with missing race (2.2%)",
                "medications": "24 med features -> binary used (any of Steady/Up/Down)",
                "diagnoses": "diag_1/2/3 ICD-9 grouped into diabetes/circulatory/respiratory/"
                             "digestive/injury/genitourinary/neoplasms/other/missing",
                "age": "bracket midpoint, numeric",
                "note": "Same patient can appear in multiple encounters (101,766 encounters, "
                        "71,518 patients); encounters treated as independent rows.",
            },
        },
        "results": results,
        "best_model_by_roc_auc": best_name,
        "fairness_best_model": fairness,
        "shap_top10": shap_top,
        "shap_model": shap_model,
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Wrote metrics.json")

    # ---- Plots ----
    plots = ROOT / "plots"
    plots.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    cm = np.array(results[best_name]["confusion_matrix"])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion matrix — {best_name} (test n={len(y_test)})")
    ax.set_xlabel("Predicted (0 = not readmitted <30d, 1 = readmitted <30d)")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(plots / "confusion_matrix.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC {results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall, <30d)")
    ax.set_title("ROC curves — readmitted within 30 days")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "roc_curves.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    feats = [s["feature"] for s in shap_top][::-1]
    vals = [s["mean_abs_shap"] for s in shap_top][::-1]
    ax.barh(feats, vals)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top 10 drivers — {shap_model} (300-row test sample)")
    fig.tight_layout()
    fig.savefig(plots / "shap_top_drivers.svg")
    plt.close(fig)
    print("Wrote plots")


if __name__ == "__main__":
    main()
