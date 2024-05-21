import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('bank-additional.csv', delimiter=';')
df.rename(columns={'y':'deposit'}, inplace=True)

# Data Exploration
print("First 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
print("\nDimensions of the dataset:")
print(df.shape)
print("\nColumn names:")
print(df.columns)
print("\nData types:")
print(df.dtypes)
print("\nNumber of different data types:")
print(df.dtypes.value_counts())
print("\nInformation about the dataset:")
print(df.info())
print("\nChecking for duplicates:")
print(df.duplicated().sum())
print("\nMissing values:")
print(df.isna().sum())

# Separate categorical and numerical columns
cat_cols = df.select_dtypes(include='object').columns
num_cols = df.select_dtypes(exclude='object').columns

# Data Visualization
df.hist(figsize=(10,10), color='#cc5500')
plt.show()

for feature in cat_cols:
    plt.figure(figsize=(5,5))
    sns.countplot(x=feature, data=df, palette='Wistia')
    plt.title(f'Bar Plot of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()

df.plot(kind='box', subplots=True, layout=(2,5), figsize=(20,10), color='#7b3f00')
plt.show()

# Removing outliers
column = df[['age', 'campaign', 'duration']]
q1 = np.percentile(column, 25)
q3 = np.percentile(column, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df[['age', 'campaign', 'duration']] = column[(column > lower_bound) & (column < upper_bound)]

df.plot(kind='box', subplots=True, layout=(2,5), figsize=(20,10), color='#808000')
plt.show()

# Correlation analysis and dropping highly correlated columns
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
print(corr)
corr = corr[abs(corr) >= 0.90]
print(corr)

high_corr_cols = ['emp.var.rate', 'euribor3m', 'nr.employed']
df1 = df.drop(high_corr_cols, axis=1)

# Encoding categorical columns
lb = LabelEncoder()
df_encoded = df1.apply(lb.fit_transform)

# Splitting data into features and target variable
x = df_encoded.drop('deposit', axis=1)
y = df_encoded['deposit']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# Model building
dt = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=10)
dt.fit(x_train, y_train)

# Model evaluation
def eval_model(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy_Score', acc)
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix\n', cm)
    print('Classification Report\n', classification_report(y_test, y_pred))

def mscore(model):
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print('Training Score', train_score)
    print('Testing Score', test_score)

ypred_dt = dt.predict(x_test)
print(ypred_dt)
eval_model(y_test, ypred_dt)


# Visualizing Decision Tree
cn = ['no', 'yes']
fn = x_train.columns
plot_tree(dt, feature_names=fn, class_names=cn, filled=True)
plt.show()