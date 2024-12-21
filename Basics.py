#Importing the pandas library

import pandas as pd
import numpy as np

# Creating pandas DataFrame

# Importing the diabetes data

from sklearn.datasets import load_diabetes

diabetes_dataset = load_diabetes()

diabetes_df = pd.DataFrame(diabetes_dataset.data, columns= diabetes_dataset.feature_names) # feature_names are column names

print(diabetes_df.head()) # Prints first five rows of the DataFrame

print(diabetes_df.shape) # No. of rows and columns in the DataFrame

# Importing Data from a csv file

diabetes_df1 = pd.read_csv('diabetes_prediction.csv')
print(type(diabetes_df1))
print(diabetes_df1.head)
print(diabetes_df1.shape)

# Loading the excel file to pd DataFrame pd.read_excel('file path')

# Exporting a DataFrame to a csv file

diabetes_df.to_csv('diabetes.csv')

# Creating a DataFrame with random values

random_df = pd.DataFrame(np.random.rand(20,10)) # Values between 0 and 1



