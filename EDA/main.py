# =====================================================> Phase 3 — Exploratory Data Analysis <==================================

# What EDA Actually Is
# Before building any model, you need to understand your data deeply. EDA is that process — you're asking questions and letting the data answer them visually and statistically.
# Most beginners skip EDA or do it superficially. That's why their models underperform. The patterns you find in EDA directly tell you:

# Which features matter
# Which features are useless
# What relationships exist between variables
# Where the data is lying to you
# What feature engineering to do in Phase 4

# Setup — Libraries You'll Use:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#========================================================>1.Univariate Analysis<===================================================

# Uni = one. You analyze one variable at a time in complete isolation. Goal is to understand its distribution, range, and shape.
# ----------------->For Numeric Columns
df = pd.read_csv('Titanic.csv')
df['Age'].describe()
# count, mean, std, min, 25%, 50%, 75%, max
# This single line tells you the entire shape of the column

# More detailed stats

# print(df['Age'].mean())
# print(df['Age'].median())
# print(df['Age'].std())
# print(df['Age'].skew())     # how asymmetric the distribution is
# print(df['Age'].kurt())     # how heavy the tails are

# *********************Hidden knowledge — skewness:

# Skew = 0 → perfectly symmetric (normal distribution)
# Skew > 0 → right skewed (tail pulls right, most values are low)
# Skew < 0 → left skewed (tail pulls left, most values are high)
# Skew > 1 or < -1 → seriously skewed, consider log transform before modeling


# Histogram — shows distribution shape
# df['Age'].hist(bins=30)
# plt.title('Age Distribution')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()

# # KDE plot — smoother version of histogram
# df['Age'].plot(kind='kde')
# plt.title('Age Density')
# plt.show()

# # Both together — the best combo
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# df['Age'].hist(bins=30, ax=axes[0])
# axes[0].set_title('Histogram')
# df['Age'].plot(kind='kde', ax=axes[1])
# axes[1].set_title('KDE Plot')
# plt.show()

# ***********************Hidden knowledge — what you're actually looking for:

# Is the target class balanced or imbalanced?
# Are there dominant categories that will bias the model?
# Are there rare categories with so few samples they'll cause problems?

# ==================================================================>2.Bivariate Analysis <==============================================
# Bi = two. You analyze the relationship between two variables. This is where you start finding patterns that matter for prediction.

# ---------------------->Numeric vs Numeric<-----------------

# Scatter plot — the fundamental bivariate plot
# plt.scatter(df['Age'], df['Fare'])
# plt.xlabel('Age')
# plt.ylabel('Fare')
# plt.title('Age vs Fare')
# plt.show()

# # Seaborn version with more info
# sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived')
# # hue adds a third variable as color — powerful
# plt.show()

# # Correlation coefficient — single number summarizing relationship
# df['Age'].corr(df['Fare'])
# # 1.0 = perfect positive, -1.0 = perfect negative, 0 = no linear relationship

# # Regression line on scatter — shows trend
# sns.regplot(data=df, x='Age', y='Fare')
# plt.show()

# *******************Hidden knowledge — correlation traps:

# Correlation only measures linear relationships
# Two variables can be strongly related but have correlation = 0 if the relationship is curved
# Always plot first, then check correlation — never trust correlation alone

# ------------------->Numeric vs Categorical<-----------------

# Box plot — shows distribution of numeric per category
# sns.boxplot(data=df, x='Pclass', y='Fare')
# plt.title('Fare by Passenger Class')
# plt.show()

# # Violin plot — like boxplot but shows full distribution shape
# sns.violinplot(data=df, x='Pclass', y='Age')
# plt.title('Age Distribution by Class')
# plt.show()

# # Bar plot of means
# df.groupby('Pclass')['Fare'].mean().plot(kind='bar')
# plt.title('Average Fare by Class')
# plt.show()

# # The most important bivariate — feature vs target
# sns.boxplot(data=df, x='Survived', y='Age')
# plt.title('Age vs Survival')
# plt.show()
# If the boxes are at different heights — Age matters for prediction
# If they overlap completely — Age might not be useful

# -----------------> Categorical vs Categorical<------------------

# Cross tabulation — counts of combinations
# pd.crosstab(df['Sex'], df['Survived'])

# # As percentages
# pd.crosstab(df['Sex'], df['Survived'], normalize='index') * 100
# # Shows survival rate per gender

# # Visualize with heatmap
# ct = pd.crosstab(df['Sex'], df['Survived'])
# sns.heatmap(ct, annot=True, fmt='d', cmap='Blues')
# plt.title('Gender vs Survival')
# plt.show()

# # Count plot with hue
# sns.countplot(data=df, x='Pclass', hue='Survived')
# plt.title('Survival by Class')
# plt.show()

# ============================================================> 3.Multivariate Analysis <=========================================================

# Multi = many. Analyzing 3 or more variables simultaneously to find complex patterns.

# # Scatter with 3 variables (x, y, color)
# sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived', style='Sex', size='Pclass')
# plt.title('Age vs Fare colored by Survival')
# plt.show()

# # FacetGrid — same plot repeated per category
# g = sns.FacetGrid(df, col='Pclass', row='Sex')
# g.map(plt.hist, 'Age', bins=20)
# plt.show()
# Now you see Age distribution for every combination of class and gender

# ==========================================================> 4.Distribution Analysis <================================
# Understanding the shape of your data's distribution is critical because many ML algorithms assume normality.

# QQ Plot — tests if data is normally distributed
# from scipy import stats
# stats.probplot(df['Fare'], plot=plt)
# plt.title('QQ Plot — Fare')
# plt.show()
# # If points follow the diagonal line → normally distributed
# # If they curve off → not normal (common with Fare, Age, Income)

# # Distribution with normal curve overlay
# from scipy.stats import norm
# mu, std = df['Age'].dropna().mean(), df['Age'].dropna().std()
# x = np.linspace(df['Age'].min(), df['Age'].max(), 100)

# df['Age'].hist(bins=30, density=True, alpha=0.7)
# plt.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2)
# plt.title('Age Distribution vs Normal Curve')
# plt.show()

# # Check skewness for all numeric columns at once
# df.select_dtypes(include=np.number).skew().sort_values(ascending=False)

# -------------->Hidden knowledge — why distribution shape matters:

# Linear regression assumes residuals are normally distributed
# Highly skewed features can give too much weight to extreme values
# Log transform fixes right skew: np.log1p(df['Fare'])
# Square root transform is gentler: np.sqrt(df['Fare'])

# ==================================================>5. Box Plot (Deep Dive)<===========================
# Box plots pack enormous information into one visual. Most people read them superficially.
# What each part tells you:
# ─────────────────────────
#     |         ← outliers (individual points beyond whiskers)
#     |
#   ──┬──       ← upper whisker (Q3 + 1.5*IQR)
#   │   │
#   │   │       ← box spans Q1 to Q3 (middle 50% of data)
#   ├───┤       ← median line (Q2)
#   │   │
#   ──┴──       ← lower whisker (Q1 - 1.5*IQR)
#     |
#     # |         ← outliers below

# Multiple box plots at once — great for comparing features
# df[['Age', 'SibSp', 'Parch']].plot(kind='box')
# plt.title('Distribution Comparison')
# plt.show()

# ============================================>6. Histograms (Deep Dive)<=================================
# Single histogram
# df['Age'].hist(bins=30)

# # All numeric columns at once
# df.hist(figsize=(12, 10), bins=30)
# plt.tight_layout()
# plt.show()


# # Overlapping histograms — compare distribution between groups
# df[df['Survived']==1]['Age'].hist(bins=30, alpha=0.5, label='Survived')
# df[df['Survived']==0]['Age'].hist(bins=30, alpha=0.5, label='Died')
# plt.legend()
# plt.title('Age Distribution by Survival')
# plt.show()
# ------------------>Hidden knowledge — bin count matters:

# Too few bins → hides the real shape
# Too many bins → noise looks like signal
# Rule of thumb: bins = sqrt(n_rows) is a safe default
# Always try multiple bin counts before concluding anything


# =========================================>7. Correlation Matrix<====================================
# Shows pairwise correlation between all numeric columns simultaneously.
# python# Compute
# corr_matrix = df.corr(numeric_only=True)
# print(corr_matrix)

# # Visualize as heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(
#     corr_matrix,
#     annot=True,          # show numbers
#     fmt='.2f',           # 2 decimal places
#     cmap='coolwarm',     # red=positive, blue=negative
#     center=0,            # white at 0
#     square=True,
#     linewidths=0.5
# )
# plt.title('Correlation Matrix')
# plt.show()

# # Only show correlations with the target
# corr_matrix['Survived'].sort_values(ascending=False)
# ------------------->Hidden knowledge — what to look for:

# High correlation with target → useful feature
# High correlation between two features (>0.85) → multicollinearity, drop one of them
# Near-zero correlation with everything → possibly useless feature
# Correlation matrix only shows linear relationships — nonlinear relationships won't appear here


# ==============================>8. Pair Plot<================================
# Scatter plot for every pair of numeric variables at once. The most information-dense EDA plot.
# python# Basic pair plot
# sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']])
# plt.show()

# # With target as color
# sns.pairplot(
#     df[['Age', 'Fare', 'Pclass', 'Survived']],
#     hue='Survived',
#     diag_kind='kde',        # diagonal shows distribution
#     plot_kws={'alpha': 0.5}
# )
# plt.show()
# ---------------------->Hidden knowledge:

# Diagonal shows each variable's own distribution (histogram or KDE)
# Off-diagonal shows scatter between every pair
# If two classes (survived/died) separate cleanly in any scatter → that feature is powerful
# On large datasets (>50 columns) pair plots become unreadable — use correlation matrix instead


# ===============================> 9. Heatmaps <==================================
# # Correlation heatmap — covered above

# # Pivot table heatmap — great for categorical vs categorical vs numeric
# pivot = df.pivot_table(values='Fare', index='Sex', columns='Pclass', aggfunc='mean')
# sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd')
# plt.title('Average Fare by Gender and Class')
# plt.show()

# # Missing value heatmap
# sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
# plt.title('Missing Value Map')
# plt.show()
# # Yellow = missing, purple = present
# # Shows you patterns in missingness — is it random or systematic?

# ==============================>10. Feature Relationships<===========================================

# Going deeper than correlation — understanding how features interact with the target.
# python# Mean target rate per category — powerful for classification
# df.groupby('Sex')['Survived'].mean()
# # female: 0.74, male: 0.19 → Sex is clearly important

# df.groupby('Pclass')['Survived'].mean()
# # 1: 0.63, 2: 0.47, 3: 0.24 → Pclass is important and ordinal

# # Point plot — shows mean with confidence interval
# sns.pointplot(data=df, x='Pclass', y='Survived', hue='Sex')
# plt.title('Survival Rate by Class and Gender')
# plt.show()

# # Binning a numeric to see relationship with target
# df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100],
#                        labels=['Child','Teen','Adult','Middle','Senior'])
# df.groupby('AgeBin')['Survived'].mean().plot(kind='bar')
# plt.title('Survival Rate by Age Group')
# plt.show()


# ============================>11. Detecting Bias<====================================
# Real world datasets carry human bias. If you don't find it, your model learns it and amplifies it.
# python# Class imbalance — is target balanced?
# df['Survived'].value_counts(normalize=True) * 100
# # 61.6% died, 38.4% survived — imbalanced
# # A model that always predicts "died" gets 61.6% accuracy doing nothing

# # Representation bias — are groups equally represented?
# df.groupby('Sex')['Survived'].agg(['count', 'mean'])

# # Selection bias — who is in the dataset?
# df['Pclass'].value_counts(normalize=True) * 100
# # If 70% are 3rd class, model understands 3rd class well but not 1st

# # Feature bias — does a feature encode protected attributes?
# # e.g. 'neighborhood' might be a proxy for race
# # You have to know your domain to catch these

# # Distribution shift check — train vs test distribution
# X_train, X_test, _, _ = train_test_split(df, df['Survived'], test_size=0.2, random_state=42)
# print("Train gender split:", X_train['Sex'].value_counts(normalize=True).values)
# print("Test gender split:", X_test['Sex'].value_counts(normalize=True).values)
# # Should be similar — if very different, your split is biased
# Hidden knowledge — why bias detection matters:

# A biased model isn't just unfair — it's also less accurate on underrepresented groups
# Class imbalance directly causes models to ignore the minority class
# You can't fix bias you haven't found — EDA is your only chance before it bakes into the model

# ===============================>Full EDA Template<======================================
# This is the checklist you run on every new dataset going forward:
# pythonimport pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats

# def full_eda(df, target):
#     print("=" * 50)
#     print("SHAPE:", df.shape)
#     print("=" * 50)

#     # 1. Basic info
#     print(df.info())
#     print(df.describe())

#     # 2. Missing values
#     missing = df.isnull().sum()
#     print("\nMissing:\n", missing[missing > 0])

#     # 3. Target distribution
#     print("\nTarget distribution:")
#     print(df[target].value_counts(normalize=True) * 100)

#     # 4. Numeric distributions
#     numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
#     df[numeric_cols].hist(bins=30, figsize=(14, 10))
#     plt.tight_layout()
#     plt.show()

#     # 5. Skewness
#     print("\nSkewness:")
#     print(df[numeric_cols].skew().sort_values(ascending=False))

#     # 6. Correlation matrix
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f',
#                 cmap='coolwarm', center=0)
#     plt.title('Correlation Matrix')
#     plt.show()

#     # 7. Feature vs target
#     for col in numeric_cols:
#         if col != target:
#             sns.boxplot(data=df, x=target, y=col)
#             plt.title(f'{col} vs {target}')
#             plt.show()

#     # 8. Categorical vs target
#     cat_cols = df.select_dtypes(include='object').columns.tolist()
#     for col in cat_cols:
#         sns.countplot(data=df, x=col, hue=target)
#         plt.title(f'{col} vs {target}')
#         plt.xticks(rotation=45)
#         plt.show()

# full_eda(df, target='Survived')