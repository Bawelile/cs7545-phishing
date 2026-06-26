import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 1 - Load data
train_df = pd.read_csv("data/processed/train.csv")
val_df   = pd.read_csv("data/processed/val.csv")
test_df  = pd.read_csv("data/processed/test.csv")

print("Train:", len(train_df), "rows")
print("Val:  ", len(val_df), "rows")
print("Test: ", len(test_df), "rows")

# Step 2 - Extract TF-IDF features
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
X_train = vectorizer.fit_transform(train_df["text_combined"])
X_val   = vectorizer.transform(val_df["text_combined"])
X_test  = vectorizer.transform(test_df["text_combined"])

y_train = train_df["label"]
y_val   = val_df["label"]
y_test  = test_df["label"]

print("TF-IDF features extracted successfully")
print("X_train shape:", X_train.shape)

# Step 3 - Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=100),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    print(f"{name}: {score:.4f}")

# Step 4 - Evaluate best model on test set
print("\n--- Final Evaluation on Test Set ---")
best_model = models["Logistic Regression"]
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred,
      target_names=["Legitimate", "Human Phishing", "AI Phishing"]))
