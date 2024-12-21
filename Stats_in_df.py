# Statistical Measures

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

diabetes_dataset = load_diabetes()

diabetes_df = pd.DataFrame(diabetes_dataset.data, columns= diabetes_dataset.feature_names)

# Number of values in a column
print(diabetes_df.count())

# Mean values - column wise
print(diabetes_df.mean())

# Standard Deviation - column wise
print(diabetes_df.std())

# Minimum Value - column wise
print(diabetes_df.min())

# Maximum Value - column wise
print(diabetes_df.max())

# All the statistical measures
print(diabetes_df.describe())
