# Manipulate a DataFrame

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

diabetes_dataset = load_diabetes()

diabetes_df = pd.DataFrame(diabetes_dataset.data, columns= diabetes_dataset.feature_names)

# Adding a column to a DataFrame
diabetes_df['Insulin'] = np.random.randint(0, 10, size=len(diabetes_df))
print(diabetes_df.head())
#
# Removing a row from a DataFrame
diabetes_df.drop(index = 0, axis= 0, inplace= True)
print(diabetes_df.head())

# Removing a column from a DataFrame
diabetes_df.drop(columns='bmi', axis = 1, inplace= True)
print(diabetes_df.head())

# Locating a row based on the index
print(diabetes_df.iloc[2])

# Locating a particular column
print(diabetes_df.iloc[:, 5]) # Locating the 6th column
print(diabetes_df.iloc[:, -1]) # Locating the last column

# Correlation
print(diabetes_df.corr())

