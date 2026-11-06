
# HOUSE PRICE PREDICTION USING LINEAR REGRESSION

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 55)
print("       HOUSE PRICE PREDICTION PROJECT")
print("       USING LINEAR REGRESSION")
print("=" * 55)

# 2. Load Housing Dataset
print("\nLoading Housing Dataset...")

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("\nDataset Loaded Successfully!")
print("\nFirst Five Records:")
print(df.head())

# 3. Dataset Information
print("\nDataset Information:")
print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

print("\nDataset Description:")
print(df.describe())

# 4. Check Missing Values
print("\nChecking Missing Values:")
print(df.isnull().sum())

# 5. Separate Features and Target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

print("\nFeatures and Target Separated Successfully!")

# 6. Split Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nData Splitting Completed!")
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# 7. Create Linear Regression Model
model = LinearRegression()

# 8. Train the Model
print("\nTraining Linear Regression Model...")

model.fit(X_train, y_train)

print("Model Trained Successfully!")

# 9. Predict House Prices
y_pred = model.predict(X_test)

print("\nHouse Price Predictions:")
results = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": y_pred[:10]
})

print(results.round(3))

# 10. Evaluate Model Using RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results:")
print("Mean Squared Error (MSE):", round(mse, 4))
print("Root Mean Squared Error (RMSE):", round(rmse, 4))
print("R2 Score:", round(r2, 4))

# 11. Visualize Actual vs Predicted Prices
plt.figure(figsize=(9, 6))

plt.scatter(y_test, y_pred, alpha=0.4)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("House Price Prediction Using Linear Regression")

plt.tight_layout()
plt.show()

# 12. Save Prediction Results
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

results.to_csv("house_price_predictions.csv", index=False)

print("\nPrediction Results Saved Successfully!")
print("File Name: house_price_predictions.csv")

# 13. Final Result
print("\n" + "=" * 55)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 55)
