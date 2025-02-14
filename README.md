# ML-CAT-1
ML CAT 1 Using PaySim - Mobile Money Fraud Dataset
# Instructions:
*Gather relevant data from various credible sources that align with the problem identified in Part A. Perform data exploration to understand the structure, patterns, and key insights within the dataset. Finally, carry out data preprocessing steps such as cleaning, handling missing values, encoding categorical data, and normalizing features to prepare the data for model development.*

The following are the codes for this Assignment
# 1. DATA EXPLORATION
## Loading the Dataset

import pandas as pd
df = pd.read_csv(r"C:\Users\Abigail\Downloads\PaySim.csv")
df.head()

#The last 5 records on the Dataset
df = pd.read_csv(r"C:\Users\Abigail\Downloads\PaySim.csv")
df.tail()

##Checking the Data Structure
#Viewing column names,data types and sample values
print(df.info())
print(df.describe())

## Identify target variable
#In this case we have the columns isFraud where 0 is a legitimate transcation and 1 a fraudulent transaction.
print(df["isFraud"].value_counts(normalize=True)) 

* #Checking the Target Variable Distribution.
#Count the number of Fraud vs Legitimate Transactions
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load the dataset
df = pd.read_csv(r"C:\Users\Abigail\Downloads\PaySim.csv")

#Count the Fraud and Legitimate transactions
fraud_counts = df["isFraud"].value_counts()
print(fraud_counts)

#Percentage Dsitribution 
fraud_percentage = df["isFraud"].value_counts(normalize=True) * 100
print(fraud_percentage) *

## Visualize the Distribution
 #Plot distribution

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load the dataset
df = pd.read_csv(r"C:\Users\Abigail\Downloads\PaySim.csv")

#Count the Fraud and Legitimate transactions
fraud_counts = df["isFraud"].value_counts()
print(fraud_counts)

#Percentage Distribution 
fraud_percentage = df["isFraud"].value_counts(normalize=True) * 100
print(fraud_percentage)

plt.figure(figsize=(6, 4))
sns.barplot(x=fraud_counts.index, y=fraud_counts.values, hue=fraud_counts.index, palette=["blue", "red"], legend=False)
plt.xticks([0, 1], ["Legitimate (0)", "Fraudulent (1)"])
plt.ylabel("Number of Transactions")
plt.title("Distribution of Fraudulent vs. Legitimate Transactions")
plt.show() 

## Correlation using a Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

#Drop categorical columns before computing correlation
df_numeric = df.select_dtypes(include=["int64", "float64"])  

#Compute correlation matrix
correlation_matrix = df_numeric.corr()

#Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

## Detect Outliers using a boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["isFraud"], y=df["amount"], color="skyblue") 
plt.xlabel("Fraudulent Transactions (0 = Legit, 1 = Fraud)")
plt.ylabel("Transaction Amount")
plt.title("Boxplot of Transaction Amounts for Fraudulent and Legitimate Transactions")
plt.show()

# 2. DATA PREPROCESSING

## Data Cleaning

df.info()
df.describe().T

## Handling missing values
print(df.isnull().sum())

## Data Cleaning
#Removing columns that won't be used in modeling, in this case nameOrig and nameDest

import pandas as pd

#Load dataset
df = pd.read_csv(r"C:\Users\Abigail\Downloads\PaySim.csv")

#Removing nameOrig and nameDest
df_cleaned = df.drop(columns=["nameOrig", "nameDest"])

#Check for missing values
print(df_cleaned.isnull().sum())  

#Fill missing values for numerical columns
numerical_columns = df_cleaned.select_dtypes(include=["float64", "int64"]).columns
df_cleaned[numerical_columns] = df_cleaned[numerical_columns].fillna(df_cleaned[numerical_columns].median())

#Filling the non-numeric columns with the Most frequent value
non_numerical_columns = df_cleaned.select_dtypes(exclude=["float64", "int64"]).columns
df_cleaned[non_numerical_columns] = df_cleaned[non_numerical_columns].fillna(df_cleaned[non_numerical_columns].mode().iloc[0])

#Display the cleaned dataset
print(df_cleaned.head())

## Encoding with Categorical Variables

df_encoded = pd.get_dummies(df_cleaned, columns=["type"], drop_first=True)  
print(df_encoded.head())

## Feature Normalization

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Select numerical features for scaling 
numerical_features = df_encoded.select_dtypes(include=["float64", "int64"]).columns

#Initialize Min-Max Scaler
scaler = MinMaxScaler()

#Apply scaling to the numerical features
df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

#Display the normalized dataset
print(df_encoded.head())

#Split the data into features (X) and target (y)
X = df_encoded.drop(columns=["isFraud", "isFlaggedFraud"])  
y = df_encoded["isFraud"]  # Target variable

#Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Check shapes of the splits
print(f"Training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Test set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")








