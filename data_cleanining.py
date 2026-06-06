# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("Titanic.csv")

print("===== Dataset Information =====")
print(df.info())

print("\n===== Statistical Summary =====")
print(df.describe())

print("\n===== Missing Values =====")
print(df.isnull().sum())


# Age -> Median Imputation
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fare -> Mean Imputation
if 'Fare' in df.columns:
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)

# Embarked -> Mode Imputation
if 'Embarked' in df.columns:
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Cabin has too many missing values, drop it
if 'Cabin' in df.columns:
    df.drop('Cabin', axis=1, inplace=True)

print("\n===== Missing Values After Cleaning =====")
print(df.isnull().sum())


# Label Encoding for Sex
le = LabelEncoder()

if 'Sex' in df.columns:
    df['Sex'] = le.fit_transform(df['Sex'])
    # male=1, female=0 (or vice versa)

# One Hot Encoding for Embarked
if 'Embarked' in df.columns:
    embarked_encoded = pd.get_dummies(
        df['Embarked'],
        prefix='Embarked',
        drop_first=True
    )

    df = pd.concat([df, embarked_encoded], axis=1)
    df.drop('Embarked', axis=1, inplace=True)

print("\n===== Encoded Dataset Sample =====")
print(df.head())


plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=30, kde=True)

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Count")

plt.show()


df.to_csv("Titanic_Cleaned.csv", index=False)

print("\nCleaned dataset saved as 'Titanic_Cleaned.csv'")