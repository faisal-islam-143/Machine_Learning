# ====================================================================> Phase 4 — Feature Engineering <=========================================
# What Feature Engineering Actually Is:

# Feature engineering is the process of creating, transforming, and selecting features to give your model better information to learn from.
# Raw data is almost never in the best form for a model. A date column is useless as a string — but extracting "is it a weekend?" or "which hour?"
#  from it can be extremely powerful.
# The dirty secret of ML competitions: the winner is almost never the person with the best model. It's the person with the best features. 
# A simple Logistic Regression with great features beats XGBoost with raw features most of the time.

# ========================================> 1. Feature Creation <================================================================================

# Creating entirely new columns from existing ones using domain knowledge or mathematical combinations.

import pandas as pd
import numpy as np

# # Combining existing features into something more meaningful
# df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# # +1 for the passenger themselves
# # A single number now captures total family size — more useful than two separate columns

# # Is the passenger alone?
# df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
# # Binary feature — alone passengers had different survival rates

# # Fare per person — Fare was shared among family members
# df['FarePerPerson'] = df['Fare'] / df['FamilySize']

# # Title extraction from Name — hidden signal in plain text
# df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
# print(df['Title'].value_counts())
# # Mr, Mrs, Miss, Master, Dr, Rev — survival rates differ dramatically per title

# # Simplify rare titles
# df['Title'] = df['Title'].replace(
#     ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],
#     'Rare'
# )
# df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

# *****************Hidden knowledge:
#                                  Feature creation is where domain knowledge matters most. A data scientist who understands the Titanic disaster knows that
# "women and children first" was the actual policy — so Sex and Age are going to be powerful,
# and combining them (IsWomanOrChild) might be even better than either alone.

# df['IsWomanOrChild'] = ((df['Sex'] == 'female') | (df['Age'] < 12)).astype(int)
# df.groupby('IsWomanOrChild')['Survived'].mean()
# You'll see a dramatic difference

# =====================================================> 2. Feature Selection <=================================================
# Not all features help. Some are noise, some are redundant, some actively hurt performance. Feature selection removes the bad ones.
# ------------------------->Method 1 — Filter Methods (Statistical)<----------------------------