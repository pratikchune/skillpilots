
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "customer_churn_dataset.csv"

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading Customer Churn Dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)
print(df.head())

# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()

# Remove duplicate records
df = df.drop_duplicates()

# Target column
target = "Churn"

if target not in df.columns:
    raise ValueError("Churn column not found in dataset!")

# Remove rows with missing target values
df = df.dropna(subset=[target])

# Remove customer ID because it is not useful for prediction
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# Separate features and target
X = df.drop(columns=[target])
y = df[target].astype(str).str.strip()

if y.empty:
    raise ValueError("Target column is empty after filtering.")

# Identify the positive class (churn)
classes = [str(c) for c in y.unique()]
if len(classes) < 2:
    raise ValueError(
        "Target column must contain at least two unique classes to train a model."
    )

positive_label = next(
    (c for c in classes if c.lower() in ["yes", "true", "1", "churn"]),
    sorted(classes, key=lambda label: label.lower())[-1],
)
negative_label = next(c for c in classes if c != positive_label)

print("\nPositive Class:", positive_label)
print("Negative Class:", negative_label)

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["int64", "float64", "int32", "float32"]
).columns.tolist()

# Numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

# Combine preprocessing while allowing datasets with only one feature type.
transformers = []
if numeric_features:
    transformers.append(("num", numeric_transformer, numeric_features))
if categorical_features:
    transformers.append(("cat", categorical_transformer, categorical_features))

if not transformers:
    raise ValueError("No usable feature columns were found for training.")

preprocessor = ColumnTransformer(transformers=transformers)

# ==========================================
# 3. SELECT MACHINE LEARNING MODEL
# ==========================================

model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# ==========================================
# 4. SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# ==========================================
# 5. TRAIN THE MODEL
# ==========================================

print("\nTraining Logistic Regression Model...")

model.fit(X_train, y_train)

print("Model Training Completed!")

# ==========================================
# 6. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 7. COMPUTE EVALUATION METRICS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred,
    pos_label=positive_label,
    zero_division=0
)

recall = recall_score(
    y_test, y_pred,
    pos_label=positive_label,
    zero_division=0
)

f1 = f1_score(
    y_test, y_pred,
    pos_label=positive_label,
    zero_division=0
)

print("\n========== MODEL EVALUATION ==========")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

# Save evaluation metrics for Power BI
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value": [accuracy, precision, recall, f1],
})

metrics_df.to_csv(PROJECT_DIR / "model_evaluation_metrics.csv", index=False)

# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[negative_label, positive_label]
)

print("\nConfusion Matrix:")
print(cm)

confusion_df = pd.DataFrame([
    {"Actual": negative_label, "Predicted": negative_label,
     "Count": cm[0, 0]},
    {"Actual": negative_label, "Predicted": positive_label,
     "Count": cm[0, 1]},
    {"Actual": positive_label, "Predicted": negative_label,
     "Count": cm[1, 0]},
    {"Actual": positive_label, "Predicted": positive_label,
     "Count": cm[1, 1]}
])

confusion_df.to_csv(PROJECT_DIR / "confusion_matrix.csv", index=False)

# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ==========================================
# 10. FIVE-FOLD CROSS VALIDATION
# ==========================================

print("\nPerforming Cross-Validation...")

# Use as many as 5 folds, depending on the
# number of records in the smallest class.
min_class_count = y_train.value_counts().min()
n_splits = min(5, int(min_class_count))

if n_splits < 2:
    raise ValueError("Not enough samples in each class for cross-validation.")

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(
        precision_score,
        pos_label=positive_label,
        zero_division=0
    ),
    "recall": make_scorer(
        recall_score,
        pos_label=positive_label,
        zero_division=0
    ),
    "f1": make_scorer(
        f1_score,
        pos_label=positive_label,
        zero_division=0
    )
}

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring
)

# Prepare cross-validation results
cv_data = []

for i in range(n_splits):
    cv_data.append({
        "Fold": f"Fold {i + 1}",
        "Accuracy": cv_results["test_accuracy"][i],
        "Precision": cv_results["test_precision"][i],
        "Recall": cv_results["test_recall"][i],
        "F1 Score": cv_results["test_f1"][i]
    })

# Add average results
cv_data.append({
    "Fold": "Average",
    "Accuracy": np.mean(cv_results["test_accuracy"]),
    "Precision": np.mean(cv_results["test_precision"]),
    "Recall": np.mean(cv_results["test_recall"]),
    "F1 Score": np.mean(cv_results["test_f1"])
})

cv_df = pd.DataFrame(cv_data)

cv_df.to_csv(PROJECT_DIR / "cross_validation_results.csv", index=False)

print("\nCross-Validation Results:")
print(cv_df.round(4).to_string(index=False))

# ==========================================
# 11. SAVE TEST PREDICTIONS
# ==========================================

predictions_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

# Add churn probability
positive_index = list(model.classes_).index(positive_label)

predictions_df["Churn_Probability"] = model.predict_proba(
    X_test
)[:, positive_index]

predictions_df.to_csv(PROJECT_DIR / "test_predictions.csv", index=False)

# ==========================================
# 12. SAVE TRAINED MODEL
# ==========================================

joblib.dump(model, PROJECT_DIR / "churn_model.pkl")

# ==========================================
# 13. FINAL OUTPUT
# ==========================================

print("\n===================================")
print("PROJECT COMPLETED SUCCESSFULLY!")
print("===================================")

print("\nGenerated Files:")
print("1. model_evaluation_metrics.csv")
print("2. cross_validation_results.csv")
print("3. confusion_matrix.csv")
print("4. test_predictions.csv")
print("5. churn_model.pkl")

print("\nYou can now import the CSV files into Power BI.")
