# =======================> DATA COLLECTION <===================
# Data collection is just gathering raw data from different resources like kaggle or any online platform

import pandas as pd
df = pd.read_csv("Titanic.csv")

# here are the 7 lines are which we always executed after inducting any type of file or dataset 

df.shape
df.describe()
df.head()
df.info()
df.columns
df.dtypes
df.nunique()

# print(df.shape)
# print(df.columns)
# print(df.dtypes)
#print(df.info())
# print(df.head())
# print(df.nunique())
# print(df.describe())

# =========================> DATA TYPES <==========================

# Pandas assigns data types automatically but sometimes the assigns datatypes wrong which makes huge problem while trainging a models

#          pandas types                   meaning 
#          int64                          integers
#          float64                        decimals
#          object                         string/mixed
#          bool                           True/False
#          datetime64                     Dates
#          category                       Fixed set of values

#Here we can change the datatype of and column which are not correct according to the type like in the titanic pclass are not int its
# is category such that they play very important role later in the model training.
df.dtypes                                        # check all types

# Fix wrong types
# df['age'] = df['age'].astype(int)
# df['price'] = df['price'].astype(float)
# df['gender'] = df['gender'].astype('category')

# # String column that contains numbers — very common issue
# df['salary'] = df['salary'].str.replace(',', '').astype(float)

# # Convert to datetime
# df['date'] = pd.to_datetime(df['date'])

# df['Pclass'] = df['Pclass'].astype('category')
# print(df.info())


#==========================>Missing Values<============================

# Missing data is the common problem in the real dataset. You need to detect it, understand why its missing, then decide what to do.
# Why data goes missing:
# * User skipped a form field
# * Sensor failed
# * Data wasn't collected for that time period
# * Merge/jion left gaps

df.isnull().sum()                 # count nulls per column
print(df.isnull().sum() / len(df) * 100) # percentage missing


import missingno as msno
msno.matrix(df)                          # visual map of nulls

# Drop — only when very few rows affected
df.dropna()                              # drop any row with null
df.dropna(subset=['age', 'salary'])      # drop only if these cols are null
df.dropna(thresh=5)                      # keep rows with at least 5 non-null values


# Fill — the more useful approach
df['age'].fillna(df['age'].mean())           # mean (numeric, no outliers)
df['age'].fillna(df['age'].median())         # median (numeric, with outliers)
df['city'].fillna(df['city'].mode()[0])      # mode (categorical)
df['salary'].fillna(method='ffill')          # forward fill (time series)
df['salary'].fillna(method='bfill')          # backward fill (time series)

# Scikit-learn imputer — better for pipelines
from sklearn.impute import SimpleImputer, KNNImputer

imp = SimpleImputer(strategy='median')       # mean, median, most_frequent, constant
df_imputed = imp.fit_transform(df[['age', 'salary']])

# KNN Imputer — fills based on similar rows (more accurate)
knn_imp = KNNImputer(n_neighbors=5)
df_imputed = knn_imp.fit_transform(df)

# Rule of thumb:

# < 5% missing → drop rows or simple fill
# 5–30% missing → impute (median/KNN)
# > 30% missing → consider dropping the column entirely


#===============================================> 4. Duplicate Data <==============================================

# Duplicates silently inflate your dataset and bias your model.

# python
# df.duplicated().sum()                        # count duplicate rows
# df[df.duplicated()]                          # see the duplicate rows
# df[df.duplicated(subset=['email'])]          # duplicates based on one column

# # Remove
# df.drop_duplicates(inplace=True)
# df.drop_duplicates(subset=['email'], keep='first', inplace=True)
# Always check for duplicates after merging two datasets — that's where they appear most.

# ===============================================>5. Outliers <============================================================

# Outliers are values that are abnormally far from the rest. They can be genuine (a billionaire's salary in a dataset) or errors (age = 999).
# Three detection methods:
# pythonimport numpy as np
# import matplotlib.pyplot as plt

# # --- Method 1: Visualization ---
# df['salary'].plot(kind='box')            # boxplot — fastest visual check
# df['salary'].hist(bins=50)               # histogram

# # --- Method 2: IQR (most common) ---
# Q1 = df['salary'].quantile(0.25)
# Q3 = df['salary'].quantile(0.75)
# IQR = Q3 - Q1

# lower = Q1 - 1.5 * IQR
# upper = Q3 + 1.5 * IQR

# outliers = df[(df['salary'] < lower) | (df['salary'] > upper)]
# print(f"{len(outliers)} outliers found")

#  ------ Remove outliers
# df_clean = df[(df['salary'] >= lower) & (df['salary'] <= upper)]

#  --- Method 3: Z-Score ---
# from scipy import stats
# z_scores = np.abs(stats.zscore(df['salary']))
# df_clean = df[z_scores < 3]              # keep rows within 3 standard deviations
# When to remove vs keep:

# Errors (age=999) → remove
# Legitimate extreme values → keep, but use robust models or robust scaling


# =============================================>6. Data Cleaning <==========================================
# Catching all the messy real-world issues that don't fit neatly into the above categories.
# python# Whitespace in strings
# df['name'] = df['name'].str.strip()

# # Inconsistent casing
# df['city'] = df['city'].str.lower()
# df['city'] = df['city'].str.title()

# # Inconsistent categories — same thing spelled differently
# df['gender'].value_counts()
# # male, Male, M, m → all the same thing
# df['gender'] = df['gender'].str.lower().str.strip()
# df['gender'] = df['gender'].replace({'m': 'male', 'f': 'female'})

# # Special characters in numeric columns
# df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)

# # Rename columns — clean names are important
# df.columns = df.columns.str.lower().str.replace(' ', '_')

# # Drop useless columns
# df.drop(columns=['id', 'unnamed_0'], inplace=True)

# ======================================> 7. Feature Scaling <===================================================
# Most ML algorithms are sensitive to the scale of features. A salary column (10,000–100,000) will dominate an age column (18–65) if you don't scale.
# Algorithms that need scaling: KNN, SVM, Linear/Logistic Regression, Neural Networks, PCA
# Algorithms that don't: Decision Trees, Random Forest, XGBoost (tree-based models are scale-invariant)

# Normalization (Min-Max Scaling)
# Squeezes all values between 0 and 1.
# Formula: (x - min) / (max - min)
# Use when: you know the data has a fixed range, or you're feeding into neural networks.
# pythonfrom sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])

# Standardization (Z-Score Scaling)
# Centers data around mean=0, std=1.
# Formula: (x - mean) / std
# Use when: data is roughly normally distributed. This is the default choice for most ML algorithms.
# pythonfrom sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])

# Robust Scaling
# Uses median and IQR instead of mean and std — not affected by outliers.
# Formula: (x - median) / IQR
# Use when: your data has significant outliers you want to keep.
# pythonfrom sklearn.preprocessing import RobustScaler

# scaler = RobustScaler()
# df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])
# Critical rule: Always fit the scaler on training data only, then transform both train and test. Never fit on test data — that's data leakage (covered below).
# pythonscaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)       # transform only, never fit

# =====================================================>8. Encoding <=====================================================
# ML models need numbers. Encoding converts categorical text columns into numeric form.

# Label Encoding
# Converts each category to an integer. Simple but implies order where there may be none.
# pythonfrom sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()
# df['gender'] = le.fit_transform(df['gender'])
# # male → 1, female → 0
# Use only for: target variable (y), or genuinely ordinal data. Don't use for nominal features with no order — the model will think 2 > 1 > 0.

# One Hot Encoding
# Creates a new binary column for each category. No false ordering.
# python# Pandas way
# df = pd.get_dummies(df, columns=['city'], drop_first=True)

# # Scikit-learn way
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False, drop='first')
# encoded = ohe.fit_transform(df[['city']])
# Use for: nominal categories (color, city, brand) with no order.
# Watch out for: high cardinality columns (city with 1000 unique values → 1000 new columns). Use Feature Hashing instead.

# Ordinal Encoding
# For categories that have a meaningful order.
# pythonfrom sklearn.preprocessing import OrdinalEncoder

# oe = OrdinalEncoder(categories=[['low', 'medium', 'high']])
# df[['education']] = oe.fit_transform(df[['education']])
# # low → 0, medium → 1, high → 2
# Use for: size (S/M/L/XL), education level, satisfaction rating.

# Feature Hashing
# Converts high-cardinality categories into a fixed number of columns using hashing. Fast and memory-efficient.
# pythonfrom sklearn.feature_extraction import FeatureHasher

# fh = FeatureHasher(n_features=10, input_type='string')
# hashed = fh.fit_transform(df['city'].astype(str))
# Use for: columns with hundreds or thousands of unique categories where OHE would explode dimensionality.

# ===============================================>9. Train/Test Split <=================================================
# You never evaluate a model on data it was trained on. Split your data before any fitting happens.
# pythonfrom sklearn.model_selection import train_test_split

# X = df.drop('target', axis=1)
# y = df['target']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,          # 80% train, 20% test
#     random_state=42,        # reproducibility
#     stratify=y              # keeps class ratio in both splits (use for classification)
# )

# print(X_train.shape, X_test.shape)
# Standard splits:

# 80/20 — most common
# 70/30 — smaller datasets
# 60/20/20 — train/validation/test when tuning hyperparameters


# ==============================================>10. Data Leakage <=======================================================
# The most dangerous mistake in ML. It's when information from outside the training data bleeds into the model, causing it to look great in testing but fail completely in production.
# Common causes:
# python
# # BAD — scaling before splitting (test data influenced the scaler)
# scaler.fit(X)                           # leakage: test stats included
# X_scaled = scaler.transform(X)
# X_train, X_test = train_test_split(X_scaled)

# # GOOD — always split first, then fit on train only
# X_train, X_test = train_test_split(X)
# scaler.fit(X_train)                     # fit on train only
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)       # transform only

# # BAD — imputing before splitting
# df['age'].fillna(df['age'].mean())      # mean includes test rows

# # BAD — using a feature that contains target information
# # e.g. using 'loan_status' to predict 'default' when they mean the same thing
# # e.g. using 'discharge_date' to predict 'hospital_readmission'
# Golden rule: the test set must be completely invisible during all preprocessing. Pretend it doesn't exist until the very final evaluation.

# ======================================> 11. Data Pipelines <=====================================================================
# Instead of manually running each step, pipelines chain them together cleanly, prevent leakage, and make deployment easy.
# pythonfrom sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer

# # Define which columns are numeric vs categorical
# numeric_features = ['age', 'salary']
# categorical_features = ['city', 'gender']

# # Numeric pipeline
# numeric_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])

# # Categorical pipeline
# categorical_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('encoder', OneHotEncoder(handle_unknown='ignore'))
# ])

# # Combine both
# preprocessor = ColumnTransformer([
#     ('num', numeric_pipeline, numeric_features),
#     ('cat', categorical_pipeline, categorical_features)
# ])

# # Add model at the end
# from sklearn.linear_model import LogisticRegression

# full_pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('model', LogisticRegression())
# ])

# # Now just fit and predict — everything happens automatically
# full_pipeline.fit(X_train, y_train)
# full_pipeline.score(X_test, y_test)
# This is how professional ML code looks. No leakage, no manual steps, fully reproducible.

# Project — Titanic Survival Prediction (Data Prep Only)
# This project covers every single concept from Phase 2 in one real dataset.
# Dataset: Titanic on Kaggle or load directly:
# pythonimport pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer, KNNImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
# from sklearn.compose import ColumnTransformer

# # Load
# url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
# df = pd.read_csv(url)

# # ── Step 1: First look ──────────────────────────────────────────
# print(df.shape)
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())
# print(df.duplicated().sum())

# # ── Step 2: Drop useless columns ────────────────────────────────
# df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# # ── Step 3: Fix data types ──────────────────────────────────────
# df['Pclass'] = df['Pclass'].astype('category')

# # ── Step 4: Handle missing values ───────────────────────────────
# print(df.isnull().sum() / len(df) * 100)
# # Age: ~20% missing → impute with median
# # Embarked: 2 rows → fill with mode
# # Cabin: 77% missing → already dropped

# df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# # ── Step 5: Outlier check ───────────────────────────────────────
# Q1 = df['Fare'].quantile(0.25)
# Q3 = df['Fare'].quantile(0.75)
# IQR = Q3 - Q1
# print(f"Fare outliers: {len(df[df['Fare'] > Q3 + 1.5 * IQR])}")
# # Keep them — high fare passengers are genuine (first class)

# # ── Step 6: Clean text columns ──────────────────────────────────
# df['Sex'] = df['Sex'].str.lower().str.strip()
# df['Embarked'] = df['Embarked'].str.upper().str.strip()

# # ── Step 7: Split BEFORE any fitting ────────────────────────────
# X = df.drop('Survived', axis=1)
# y = df['Survived']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # ── Step 8: Build pipeline ──────────────────────────────────────
# numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
# categorical_features = ['Sex', 'Embarked']

# numeric_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])

# categorical_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('encoder', OneHotEncoder(handle_unknown='ignore'))
# ])

# preprocessor = ColumnTransformer([
#     ('num', numeric_pipeline, numeric_features),
#     ('cat', categorical_pipeline, categorical_features)
# ])

# # ── Step 9: Fit on train, transform both ────────────────────────
# X_train_processed = preprocessor.fit_transform(X_train)
# X_test_processed = preprocessor.transform(X_test)   # transform only

# print("Train shape:", X_train_processed.shape)
# print("Test shape:", X_test_processed.shape)
# print("Data preparation complete.")
# What this project covers:

#✅ Data Collection
#✅ Data Types
#✅ Missing Values
#✅ Duplicate Data
#✅ Outliers
#✅ Data Cleaning
#✅ Feature Scaling
#✅ Encoding
#✅ Train/Test Split
#✅ Data Leakage prevention
#✅ Data Pipelines