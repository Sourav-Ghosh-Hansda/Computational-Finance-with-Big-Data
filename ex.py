import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from keras.models import Sequential
from keras.layers import Dense

# load the data
data = pd.read_csv('MLF_GP1_CreditScore.csv')

# split the data into features and target
a = data.iloc[:, :-2]
b = data.iloc[:, -2:]

# convert categorical target variable to numerical values using one-hot encoding
encoder = OneHotEncoder()
b_one_hot = encoder.fit_transform(b).toarray()

# encode the target variable
label_encoder = LabelEncoder()
b['Rating'] = label_encoder.fit_transform(b['Rating'])

# split the data into training and testing sets
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42)

# standardize the data
scaler = StandardScaler()
a_train = scaler.fit_transform(a_train)
a_test = scaler.transform(a_test)

# Ridge regression
ridge = Ridge(alpha=0.3)
ridge.fit(a_train, b_train)
ridge_score = ridge.score(a_test, b_test)
print("Ridge Regression Accuracy:", ridge_score)

# Lasso regression
lasso = Lasso(alpha=0.05)
lasso.fit(a_train, b_train)
lasso_score = lasso.score(a_test,b_test)
print("Lasso Regression Accuracy:", lasso_score)

# Ridge logistic regression
ridge_logreg = LogisticRegression(penalty='l2', solver='lbfgs')
ridge_logreg.fit(a_train, b_train['InvGrd'])
ridge_logreg_score = ridge_logreg.score(a_test, b_test['InvGrd'])
print("Ridge Logistic Regression Accuracy:", ridge_logreg_score)

# Lasso logistic regression
lasso_logreg = LogisticRegression(penalty='l1', solver='liblinear')
lasso_logreg.fit(a_train, b_train['InvGrd'])
lasso_logreg_score = lasso_logreg.score(a_test, b_test['InvGrd'])
print("Lasso Logistic Regression Accuracy:", lasso_logreg_score)

# define the model
model = Sequential()
model.add(Dense(50, input_shape=(a_train.shape[1],), activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
model.fit(a_train, b_train['Rating'], epochs=20, batch_size=50, validation_split=0.2)

# evaluate the model
_, accuracy = model.evaluate(a_test, b_test['Rating'])
print('Neural Network Accuracy:', accuracy)
