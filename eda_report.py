
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load CSV dataset
df = pd.read_csv("sales.csv")

# Display dataset
print("First 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

df.describe().to_csv("outputs/summary_statistics.csv")

# Sales by product
sales_by_product = df.groupby("Product")["Total_Sales"].sum()
print("\nSales by Product:")
print(sales_by_product)

sales_by_product.plot(kind="bar", title="Sales by Product")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("outputs/sales_by_product.png")
plt.show()

# Sales by region
sales_by_region = df.groupby("Region")["Total_Sales"].sum()
sales_by_region.plot(kind="bar", title="Sales by Region")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("outputs/sales_by_region.png")
plt.show()

# Export dataset for Power BI
df.to_csv("outputs/cleaned_data.csv", index=False)

print("\nEDA completed successfully!")
print("Check the outputs folder.")
