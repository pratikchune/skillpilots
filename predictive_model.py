
# Task 5: Predictive Model Implementation

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 2. Load Dataset
df = pd.read_csv("sales.csv")

print("Dataset Loaded Successfully")
print(df.head())

# 3. Data Preprocessing
df["Order_Date"] = pd.to_datetime(
    df["Order_Date"], errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)

df["Unit_Price"] = pd.to_numeric(
    df["Unit_Price"], errors="coerce"
)

df["Total_Sales"] = pd.to_numeric(
    df["Total_Sales"], errors="coerce"
)

df = df.dropna(
    subset=[
        "Order_Date", "Quantity",
        "Unit_Price", "Total_Sales",
        "Category", "Region"
    ]
)

# 4. Feature Engineering
df["Month"] = df["Order_Date"].dt.month
df["Year"] = df["Order_Date"].dt.year

print("\nNew Features Created:")
print(df[["Month", "Year"]].head())

# 5. Define Target and Input Features
X = df[
    ["Quantity", "Unit_Price", "Category",
     "Region", "Month", "Year"]
]

y = df["Total_Sales"]

# 6. Encode Categorical Features
categorical_features = ["Category", "Region"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "category",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# 7. Create Regression Model
model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("regression", LinearRegression())
    ]
)

# 8. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. Train Model
model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# 10. Make Predictions
y_pred = model.predict(X_test)

print("\nActual vs Predicted Sales:")
results = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

print(results)

# 11. Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# 12. Visualize Results
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.tight_layout()
plt.show()

# 13. Save Prediction Results
results.to_csv("prediction_results.csv", index=False)

print("\nPrediction results saved successfully!")
