import pandas as pd

df = pd.read_csv("results/battery_features.csv")

print("Shape:", df.shape)
print("\nColumn info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())

print("\nCorrelation with SOH:")
print(df.corr(numeric_only=True)["SOH"].sort_values(ascending=False))

print("\nCorrelation with RUL:")
print(df.corr(numeric_only=True)["RUL"].sort_values(ascending=False))