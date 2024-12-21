import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

diabetes_dataset = load_diabetes()

diabetes_df = pd.DataFrame(diabetes_dataset.data, columns= diabetes_dataset.feature_names)

# Finding the number of rows and columns
print(diabetes_df.shape)

# First five rows in a DataFrame
print(diabetes_df.head())

# Last five rows in a DataFrame
print(diabetes_df.tail())

# Informations about DataFrame
print(diabetes_df.info())

# Find number of missing values
print(diabetes_df.isnull().sum())

# Counting the values based on the labels
print(diabetes_df.value_counts('age'))

# Group the values based on the mean
print(diabetes_df.groupby('age').mean())