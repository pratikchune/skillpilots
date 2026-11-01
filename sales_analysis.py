
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("sales_performance_dataset.csv")

# Convert date column
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Display dataset information
print(df.head())
print(df.describe())

sns.set_theme(style="whitegrid")

# 1. Sales by Category
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Category", y="Sales", estimator=sum)
plt.title("Sales by Category")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("sales_by_category.png")
plt.show()

# 2. Monthly Sales Trend
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)
monthly = df.groupby("Month", as_index=False)["Sales"].sum()

plt.figure(figsize=(10, 5))
sns.lineplot(data=monthly, x="Month", y="Sales", marker="o")
plt.title("Monthly Sales Trend")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_sales.png")
plt.show()

# 3. Sales by Region
region = df.groupby("Region")["Sales"].sum()

plt.figure(figsize=(6, 6))
plt.pie(region, labels=region.index, autopct="%1.1f%%")
plt.title("Sales Distribution by Region")
plt.savefig("sales_by_region.png")
plt.show()

# 4. Profit by Category
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Category", y="Profit", estimator=sum)
plt.title("Profit by Category")
plt.tight_layout()
plt.savefig("profit_by_category.png")
plt.show()

print("All visualizations created successfully!")
