import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("campus_crowd_dataset.xlsx")

# Show first few rows
print("Dataset Preview:")
print(df.head())

# Basic statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Plot crowd count variation by hour
plt.figure(figsize=(8,5))
plt.plot(df['Hour'], df['Crowd_Count'], marker='o')
plt.title("Crowd Count Variation by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Crowd Count")
plt.grid(True)
plt.show()
