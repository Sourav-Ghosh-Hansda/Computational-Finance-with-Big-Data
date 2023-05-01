import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV file into a pandas dataframe
df = pd.read_csv('MLF_GP1_CreditScore.csv')

# Split the dataset into training and test sets
X = df.iloc[:, :-2].values
y_investment = df.iloc[:, -2].values
y_rating = df.iloc[:, -1].values
X_train, X_test, y_train_investment, y_test_investment, y_train_rating, y_test_rating = train_test_split(
    X, y_investment, y_rating, test_size=0.2, random_state=42)

# Encode the investment grade target variable
le = LabelEncoder()
y_train_investment = le.fit_transform(y_train_investment)
y_test_investment = le.transform(y_test_investment)

# Scale the input variables
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Linear regression with Ridge regularization
reg_ridge = Ridge(alpha=0.1)
reg_ridge.fit(X_train, y_train_investment)
y_pred_ridge = reg_ridge.predict(X_test)
acc_ridge = accuracy_score(y_test_investment, y_pred_ridge.round())
print("Accuracy for linear regression with Ridge regularization: {:.2f}%".format(acc_ridge*100))

# Linear regression with Lasso regularization
reg_lasso = Lasso(alpha=0.1)
reg_lasso.fit(X_train, y_train_investment)
y_pred_lasso = reg_lasso.predict(X_test)
acc_lasso = accuracy_score(y_test_investment, y_pred_lasso.round())
print("Accuracy for linear regression with Lasso regularization: {:.2f}%".format(acc_lasso*100))

# Logistic regression with Ridge regularization
logreg_ridge = LogisticRegression(penalty='l2', solver='lbfgs')
logreg_ridge.fit(X_train, y_train_investment)
y_pred_logreg_ridge = logreg_ridge.predict(X_test)
acc_logreg_ridge = accuracy_score(y_test_investment, y_pred_logreg_ridge)
print("Accuracy for logistic regression with Ridge regularization: {:.2f}%".format(acc_logreg_ridge*100))

# Logistic regression with Lasso regularization
logreg_lasso = LogisticRegression(penalty='l1', solver='liblinear')
logreg_lasso.fit(X_train, y_train_investment)
y_pred_logreg_lasso = logreg_lasso.predict(X_test)
acc_logreg_lasso = accuracy_score(y_test_investment, y_pred_logreg_lasso)
print("Accuracy for logistic regression with Lasso regularization: {:.2f}%".format(acc_logreg_lasso*100))

# Neural network
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.1, max_iter=1000)
mlp.fit(X_train, y_train_rating)
y_pred_rating = mlp.predict(X_test)
acc_rating = accuracy_score(y_test_rating, y_pred_rating)
print("Accuracy for neural network: {:.2f}%".format(acc_rating*100))

# Evaluate the classification report for the neural network
print(classification_report(y_test_rating, y_pred_rating))
