Here is the link to google colabs 
https://colab.research.google.com/drive/1fGyNk_PVgAk190nH3px52nOvy32vL0rI?usp=sharing
 
# Step 1: Import libraries and load dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset from URL
url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
df = pd.read_csv(url)
# Step 2: Quality investigation
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
# Step 3: Exploratory data analysis
print("Iris dataset has", df.shape[0], "rows and", df.shape[1], "columns.")
print("Summary statistics:\n", df.describe())
# Step 4: Feature selection
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
corr = X.corr()
corr_threshold = 0.8
high_corr_features = np.where(corr > corr_threshold)
high_corr_features = [(corr.columns[x], corr.columns[y]) for x, y in zip(*high_corr_features) if x != y and x < y]
X.drop(columns=[col[1] for col in high_corr_features], inplace=True)
# Step 5: Mutual information analysis
encoder = LabelEncoder()
for column in X.select_dtypes(include=[object]).columns:
    X[column] = encoder.fit_transform(X[column])
mutual_info = mutual_info_classif(X, y, discrete_features='auto', random_state=1)
mutual_info = pd.Series(mutual_info, index=X.columns)
print("Mutual information with target variable:\n", mutual_info.sort_values(ascending=False))
# Step 6: Data preprocessing
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X = pd.get_dummies(X)
# Step 7: Algorithm harness
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
logistic = LogisticRegression()
logistic.fit(X_train, y_train)
y_pred = logistic.predict(X_test)
print("Logistic Regression performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 score:", f1_score(y_test, y_pred, average='macro'))
# Step 8: Deep dive analysis
coef_df = pd.DataFrame(logistic.coef_, columns=X.columns)
print("Logistic Regression coefficients:\n", coef_df)
