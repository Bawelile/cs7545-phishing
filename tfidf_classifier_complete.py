"""
CS7545 — AI Phishing Detection
TF-IDF + Multi-Model Classifier 
Three-class: Legitimate / Human Phishing / AI Phishing

Produces:
  - results/tfidf_classification_report.csv
  - results/tfidf_confusion_matrix.csv
  - results/tfidf_model_comparison.csv
  - results/tfidf_all_predictions.csv
  - models/best_model.pkl
  - models/vectorizer.pkl
"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
from xgboost import XGBClassifier

# ── SETUP ─────────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
os.makedirs("models",  exist_ok=True)

CLASS_NAMES = ["Legitimate", "Human Phishing", "AI Phishing"]

# ── STEP 1: LOAD DATA ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading data")
print("=" * 60)

train_df = pd.read_csv("data/processed/train.csv")
val_df   = pd.read_csv("data/processed/val.csv")
test_df  = pd.read_csv("data/processed/test.csv")

print(f"Train : {len(train_df):,} rows")
print(f"Val   : {len(val_df):,} rows")
print(f"Test  : {len(test_df):,} rows")
print(f"Total : {len(train_df)+len(val_df)+len(test_df):,} rows")
print()

# Class distribution check
print("Train class distribution:")
for label, name in enumerate(CLASS_NAMES):
    count = (train_df["label"] == label).sum()
    print(f"  {name}: {count:,}")
print()

# ── STEP 2: TF-IDF FEATURE EXTRACTION ────────────────────────────────────────
print("=" * 60)
print("STEP 2 — Extracting TF-IDF features")
print("=" * 60)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),      # unigrams + bigrams
    max_features=10000,       # top 10k features
    sublinear_tf=True,        # apply log normalisation
    min_df=2,                 # ignore very rare terms
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{2,}", # minimum 2-character tokens
)

# Fit ONLY on training data — critical to prevent data leakage
X_train = vectorizer.fit_transform(train_df["text_combined"])
X_val   = vectorizer.transform(val_df["text_combined"])
X_test  = vectorizer.transform(test_df["text_combined"])

y_train = train_df["label"]
y_val   = val_df["label"]
y_test  = test_df["label"]

print(f"Feature matrix shape: {X_train.shape}")
print(f"  {X_train.shape[0]:,} training samples")
print(f"  {X_train.shape[1]:,} TF-IDF features")
print()

# ── STEP 3: TRAIN ALL THREE MODELS ───────────────────────────────────────────
print("=" * 60)
print("STEP 3 — Training models")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        multi_class="auto",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    ),
}

val_results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    val_acc   = accuracy_score(y_val, val_preds)
    val_f1    = f1_score(y_val, val_preds, average="weighted")
    val_results[name] = {
        "model":    model,
        "val_acc":  val_acc,
        "val_f1":   val_f1,
        "val_preds": val_preds
    }
    print(f"  Validation accuracy : {val_acc:.4f}")
    print(f"  Validation F1       : {val_f1:.4f}")
    print()

# ── STEP 4: SELECT BEST MODEL ─────────────────────────────────────────────────
print("=" * 60)
print("STEP 4 — Selecting best model by validation F1")
print("=" * 60)

best_name  = max(val_results, key=lambda k: val_results[k]["val_f1"])
best_model = val_results[best_name]["model"]

print(f"Best model: {best_name}")
print(f"  Val F1  : {val_results[best_name]['val_f1']:.4f}")
print(f"  Val Acc : {val_results[best_name]['val_acc']:.4f}")
print()

# Model comparison table — save to CSV
comparison_rows = []
for name, res in val_results.items():
    comparison_rows.append({
        "Model":              name,
        "Val_Accuracy":       round(res["val_acc"], 4),
        "Val_F1_Weighted":    round(res["val_f1"], 4),
        "Selected_as_Best":   name == best_name
    })

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv("results/tfidf_model_comparison.csv", index=False)
print("Saved: results/tfidf_model_comparison.csv")

# ── STEP 5: EVALUATE BEST MODEL ON TEST SET ───────────────────────────────────
print()
print("=" * 60)
print("STEP 5 — Final evaluation on held-out test set")
print("=" * 60)

y_pred      = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)
confidence  = np.max(y_pred_prob, axis=1)

test_acc = accuracy_score(y_test, y_pred)
test_f1  = f1_score(y_test, y_pred, average="weighted")

print(f"Test accuracy : {test_acc:.4f}")
print(f"Test F1       : {test_f1:.4f}")
print()

# Classification report
report = classification_report(
    y_test, y_pred,
    target_names=CLASS_NAMES,
    output_dict=True
)
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# Save classification report to CSV
report_rows = []
for cls in CLASS_NAMES:
    report_rows.append({
        "Class":     cls,
        "Precision": round(report[cls]["precision"], 4),
        "Recall":    round(report[cls]["recall"],    4),
        "F1":        round(report[cls]["f1-score"],  4),
        "Support":   int(report[cls]["support"])
    })
report_rows.append({
    "Class":     "Weighted Average",
    "Precision": round(report["weighted avg"]["precision"], 4),
    "Recall":    round(report["weighted avg"]["recall"],    4),
    "F1":        round(report["weighted avg"]["f1-score"],  4),
    "Support":   int(report["weighted avg"]["support"])
})

report_df = pd.DataFrame(report_rows)
report_df.to_csv("results/tfidf_classification_report.csv", index=False)
print("Saved: results/tfidf_classification_report.csv")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
cm_df.to_csv("results/tfidf_confusion_matrix.csv")
print("Saved: results/tfidf_confusion_matrix.csv")

# ── STEP 6: SAVE ALL PREDICTIONS ─────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 6 — Saving all predictions")
print("=" * 60)

label_to_name = {i: name for i, name in enumerate(CLASS_NAMES)}

predictions_df = test_df.copy()
predictions_df["predicted_label"]      = y_pred
predictions_df["predicted_label_name"] = [label_to_name[p] for p in y_pred]
predictions_df["actual_label_name"]    = [label_to_name[a] for a in y_test]
predictions_df["prediction_confidence"]= confidence.round(6)
predictions_df["correct"]              = y_pred == y_test.values

predictions_df.to_csv("results/tfidf_all_predictions.csv", index=False)
print("Saved: results/tfidf_all_predictions.csv")

# Quick summary
total   = len(predictions_df)
correct = predictions_df["correct"].sum()
errors  = total - correct
print(f"\nPrediction summary:")
print(f"  Total  : {total:,}")
print(f"  Correct: {correct:,}")
print(f"  Errors : {errors:,}")
print()

# ── STEP 7: SERIALISE MODEL AND VECTORIZER ────────────────────────────────────
print("=" * 60)
print("STEP 7 — Saving model and vectorizer for demo")
print("=" * 60)

with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print(f"Saved: models/best_model.pkl  ({best_name})")

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("Saved: models/vectorizer.pkl")

with open("models/model_metadata.pkl", "wb") as f:
    pickle.dump({
        "best_model_name": best_name,
        "class_names":     CLASS_NAMES,
        "val_accuracy":    val_results[best_name]["val_acc"],
        "val_f1":          val_results[best_name]["val_f1"],
        "test_accuracy":   test_acc,
        "test_f1":         test_f1,
        "n_features":      X_train.shape[1],
        "n_train_samples": X_train.shape[0],
    }, f)
print("Saved: models/model_metadata.pkl")

print()
print("=" * 60)
print("DONE — All files saved")
print("=" * 60)
print()
print("Results files:")
print("  results/tfidf_model_comparison.csv")
print("  results/tfidf_classification_report.csv")
print("  results/tfidf_confusion_matrix.csv")
print("  results/tfidf_all_predictions.csv")
print()
print("Model files for Streamlit demo:")
print("  models/best_model.pkl")
print("  models/vectorizer.pkl")
print("  models/model_metadata.pkl")
