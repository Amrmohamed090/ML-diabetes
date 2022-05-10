import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv('data\diabetes - copy.csv') 
diabetes_dataset = diabetes_dataset.drop(["patient_number"], axis=1)
X = diabetes_dataset.drop(columns = 'Diabetes', axis=1)
Y = diabetes_dataset['Diabetes']
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
Y = diabetes_dataset['Diabetes']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=5)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

classifier = LogisticRegression()
classifier.fit(X_train, Y_train)
X_train_prediction = classifier.predict(X_train)
accuracy_score(X_train_prediction, Y_train)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_train)
X_train_prediction = model.predict(X_train)
accuracy_score(X_train_prediction, Y_train)
X_test_prediction = model.predict(X_test)
accuracy_score(X_test_prediction, Y_test)
print(accuracy_score(X_test_prediction, Y_test))