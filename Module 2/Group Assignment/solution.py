# Step 1 
from sklearn.datasets import load_iris
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
#Step 2 Perform the quality investigation
# Load Iris dataset
iris = load_iris()
# Create a Pandas dataframe from the dataset
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# Add target variable to dataframe
df['target'] = iris.target
# Check for null values
print(df.isnull().sum())
# Drop duplicates
df.drop_duplicates(inplace=True)
# Check for missing values
print(df.isna().sum())
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='class')

# Step 2: Remove any unused columns
# There are no unused columns in the iris dataset

# Step 3: Check for correlation between features
corr_matrix = X.corr()
print(f"Correlation matrix:\n{corr_matrix}")

# Step 4: Remove correlated features
# There are no highly correlated features in the iris dataset
# Step 5: Check for low variance features
var_threshold = 0.1  # Set threshold for variance
variances = X.var()
low_var_cols = variances[variances < var_threshold].index.tolist()
print(f"Low variance columns:\n{low_var_cols}")
# Step 6: Remove low variance features
X.drop(columns=low_var_cols, inplace=True)
print(f"X with low variance columns removed:\n{X}")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load iris dataset
iris_df = sns.load_dataset('iris')
# Check for missing values
print(f"Missing values:\n{iris_df.isnull().sum()}\n")
# Check for duplicates
print(f"Duplicates: {iris_df.duplicated().sum()}\n")
# Explore dataset
print(f"Dataset summary:\n{iris_df.describe()}\n")
print(f"Correlation matrix:\n{iris_df.corr()}\n")
sns.pairplot(iris_df, hue='species')
plt.show()
#  Step 05 following are the sub steps which we can perform
# Step 1: Load the iris dataset
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='class')

# Step 2: Remove any unused columns
# There are no unused columns in the iris dataset

# Step 3: Check for correlation between features
corr_matrix = X.corr()
print(f"Correlation matrix:\n{corr_matrix}")

# Step 4: Remove correlated features
# There are no highly correlated features in the iris dataset

#Step 6  Conduct a mutual information analysis on the categorical variables in your dataset. This will help you determine which categorical features provide the most information about the target variable.

# Create pandas dataframe from iris data
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# Add target variable
df['target'] = iris.target
# Subset categorical columns
categorical_cols = ['target']
X_cat = df[categorical_cols]
# Compute mutual information scores
mi_scores = mutual_info_classif(X_cat, df['target'], discrete_features=True)
# Create pandas series from mutual information scores
mi_scores_series = pd.Series(mi_scores, index=X_cat.columns)
# Sort series by descending mutual information scores
mi_scores_series = mi_scores_series.sort_values(ascending=False)
# Print results
print(mi_scores_series)
# Step 6 Scale all numeric data using a suitable scaling method (such as MinMaxScaler or StandardScaler), and encode all categorical data using an appropriate encoding technique (like OneHotEncoder or LabelEncoder).
iris = load_iris()
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
iris = load_iris()
#07 Run your preprocessed data through an algorithm harness to train and test different models. This will help you identify which models perform best on your dataset
# Preprocess the data
X = iris.data
y = iris.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
encoder = LabelEncoder()
y = encoder.fit_transform(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and test different models
models = [('Logistic Regression', LogisticRegression()), ('Decision Tree', DecisionTreeClassifier()), ('Random Forest', RandomForestClassifier())]

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy: {accuracy}")
    
 #Step 8 Focus on the logistic regression algorithm and perform a deep dive analysis. Investigate the model's performance, interpret the coefficients, and identify any possible improvements that could be made.
 #Load iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scale numeric features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Fit logistic regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
# Evaluate model performance on test set
y_pred = lr_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix:\n{cm}")
coefficients = pd.DataFrame(lr_model.coef_, columns=X.columns).T
coefficients.columns = ['Coefficient 1', 'Coefficient 2', 'Coefficient 3']
print(f"\nCoefficients:\n{coefficients}")
# Identify possible improvements
# - Feature selection: remove less important features
# - Hyperparameter tuning: tune regularization parameter, solver, max_iter, etc.
# - Ensemble learning: combine with other models to improve performance
# Fit logistic regression model
lr_model = LogisticRegression(max_iter=10000)
lr_model.fit(X_train, y_train)
# Step 09
# Evaluate model performance
y_pred = lr_model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='weighted')
auc = roc_auc_score(y_test, lr_model.predict_proba(X_test), multi_class='ovr')
print(f'F1 score: {f1:.2f}')
print(f'AUC: {auc:.2f}')

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, lr_model.predict_proba(X_test)[:, 1], pos_label=2)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

# Step 5: Check for low variance features
var_threshold = 0.1  # Set threshold for variance
variances = X.var()
low_var_cols = variances[variances < var_threshold].index.tolist()
print(f"Low variance columns:\n{low_var_cols}")

# Step 6: Remove low variance features
X.drop(columns=low_var_cols, inplace=True)
print(f"X with low variance columns removed:\n{X}")

