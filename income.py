def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# 2
income_data = pd.read_csv('income.csv', header = 0)
print(income_data.head())

# 3
print(income_data.iloc[0])

# 4
income_df = pd.read_csv("income.csv", header = 0, delimiter = ", ")
print(income_df.iloc[0])

# 12
income_df["sex-int"] = income_df["sex"].apply(lambda row: 0 if row == "Male" else 1)

# 14
print(income_df["native-country"].value_counts())

# 15
income_df["country-int"] = income_df["native-country"].apply(lambda row: 0 if row == "United-States" else 1)

# 14
print(income_df["education"].value_counts())

# 15
income_df["education-int"] = income_df["education"].apply(lambda row: 0 if row == "HS-grad" else 1)

#5
labels = income_df[['income']]

# 6, 10
data = income_df[['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex-int', 'country-int', 'education-int']]

# 7
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

# 8
forest = RandomForestClassifier (random_state = 1)

# 9
forest.fit(train_data, train_labels)

# 11
print(forest.score(test_data, test_labels))





