import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv("MLF_GP1_CreditScore.csv")

# Define X (predictors) and y (response)
X = data.iloc[:, :-2].values
y_rating = data.iloc[:, -2].values
y_ig = data.iloc[:, -1].values

# Split the dataset into a training set and a test set in a 80:20 ratio
X_train, X_test, y_rating_train, y_rating_test, y_ig_train, y_ig_test = train_test_split(X, y_rating, y_ig, test_size=0.2, random_state=42)

# Apply feature scaling to the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression with Ridge (L1) Regularization
ridge_reg = Ridge(alpha=0.1)
ridge_reg.fit(X_train, y_ig_train)
y_ig_pred_ridge = ridge_reg.predict(X_test)
print("Linear Regression with Ridge (L1) Regularization:")
print(classification_report(y_ig_test, y_ig_pred_ridge))

# Linear Regression with Lasso (L2) Regularization
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_ig_train)
y_ig_pred_lasso = lasso_reg.predict(X_test)
print("Linear Regression with Lasso (L2) Regularization:")
print(classification_report(y_ig_test, y_ig_pred_lasso))

# Logistic Regression with Ridge (L1) Regularization
ridge_logreg = LogisticRegression(penalty='l1', solver='saga')
ridge_logreg.fit(X_train, y_ig_train)
y_ig_pred_ridge_logreg = ridge_logreg.predict(X_test)
print("Logistic Regression with Ridge (L1) Regularization:")
print(classification_report(y_ig_test, y_ig_pred_ridge_logreg))

# Logistic Regression with Lasso (L2) Regularization
lasso_logreg = LogisticRegression(penalty='l2')
lasso_logreg.fit(X_train, y_ig_train)
y_ig_pred_lasso_logreg = lasso_logreg.predict(X_test)
print("Logistic Regression with Lasso (L2) Regularization:")
print(classification_report(y_ig_test, y_ig_pred_lasso_logreg))

# Neural Networks Classifier
nn_clf = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.1, solver='adam', max_iter=500)
nn_clf.fit(X_train, y_rating_train)
y_rating_pred_nn = nn_clf.predict(X_test)
print("Neural Networks Classifier:")
print(classification_report(y_rating_test, y_rating_pred_nn))
