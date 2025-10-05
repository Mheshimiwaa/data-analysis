import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

#  Load and Explore the Dataset
try:
    iris_data = load_iris()
    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df["species"] = iris_data.target
    df["species"] = df["species"].map(dict(zip(range(3), iris_data.target_names)))
except Exception as e:
    print("Error loading dataset:", e)
    exit()

print("\n--- Preview of the Dataset ---")
print(df.head())

print("\n--- Dataset Information ---")
print(df.info())

print("\n--- Checking for Missing Values ---")
print(df.isnull().sum())

# Clean dataset (drop rows with missing values if any)
df.dropna(inplace=True)

#  Basic Data Analysis 
print("\n--- Statistical Summary ---")
print(df.describe())

# Group by species and compute mean of numerical features
grouped = df.groupby("species").mean()
print("\n--- Mean Values per Species ---")
print(grouped)

# Identify a simple pattern
print("\n--- Observations ---")
print("1. Iris-setosa has the smallest petal length and width on average.")
print("2. Iris-virginica has the largest measurements overall.")
print("3. The sepal dimensions vary less compared to petal dimensions.")

#  Data Visualization 
sns.set(style="whitegrid")

# Line chart (simulate a time trend using index)
plt.figure(figsize=(8, 4))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length", color="blue")
plt.title("Trend of Sepal Length across Observations")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart: average petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(x="species", y="petal length (cm)", data=df, palette="viridis")
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# Histogram: distribution of sepal width
plt.figure(figsize=(6, 4))
plt.hist(df["sepal width (cm)"], bins=15, color="orange", edgecolor="black")
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Scatter plot: sepal length vs petal length
plt.figure(figsize=(6, 4))
sns.scatterplot(
    x="sepal length (cm)",
    y="petal length (cm)",
    hue="species",
    data=df,
    palette="deep"
)
plt.title("Sepal Length vs Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.tight_layout()
plt.show()

print("\n--- Program Completed Successfully ---")
