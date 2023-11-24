# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 17:07:48 2023

@author: Thibaut R
"""

# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt


# Algorithms

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
# opening and reading the file
test_tt = pd.read_csv("C:/Users/phili/titanic_test.csv")
train_tt = pd.read_csv("C:/Users/phili/titanic_train.csv")
gender=pd.read_csv("C:/Users/phili/gender_submission.csv")
train_tt.info()
train_tt.describe()
train_tt.head(5)
#Find and count all the missing data
total = train_tt.isnull().sum().sort_values(ascending=False)
percent_1 = train_tt.isnull().sum()/train_tt.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)

#Plot to show the repartition of people who survived/not survived
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train_tt[train_tt['Sex']=='female']
men = train_tt[train_tt['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')

sns.barplot(x='Pclass', y='Survived', data=train_tt)

#Drop all the data not useful
train_tt.columns.values
train_tt=train_tt.drop(['Cabin'],axis=1)
train_tt=train_tt.drop(['PassengerId'],axis=1)
train_tt=train_tt.drop(['Name'],axis=1)
test_tt=test_tt.drop(['Cabin'],axis=1)
test_tt=test_tt.drop(['Name'],axis=1)
train_tt = train_tt.drop(['Ticket'], axis=1)
test_tt = test_tt.drop(['Ticket'], axis=1)
train_tt.columns.values
train_tt.head(5)
common_value = 'S'
data = [train_tt, test_tt]

#remplace the 2 missing values by the most common_value
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
sns.barplot(x='Pclass', y='Survived', data=train_tt)

data = [train_tt, test_tt]
#remplace the uknown age using random variable and expectecions
for dataset in data:
    mean = train_tt["Age"].mean()
    std = test_tt["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_tt["Age"].astype(int)
train_tt["Age"].isnull().sum()

data = [train_tt, test_tt]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)
#convert male and female to 0 and 1
genders = {"male": 0, "female": 1}
data = [train_tt, test_tt]
for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)
#convert embarked to 0 1 2
ports = {"S": 0, "C": 1, "Q": 2}
data = [train_tt, test_tt]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)
common_value = 'S'

#Split the different ages to categories and convert to number
data = [train_tt, test_tt]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 4
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] =4
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 6
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] =6
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] =6
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] =4
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 2
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 2
data = [train_tt, test_tt]
for dataset in data:
    dataset['Calcul']= dataset['Age']*(dataset['Sex']+1)-dataset['Pclass']

#configuration of parameters x and Y
X_train = train_tt.drop("Survived", axis=1)
Y_train = train_tt["Survived"]
X_test  = test_tt.drop("PassengerId", axis=1).copy()
Y_test=gender["Survived"]

#test with logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)
# calculate and print results
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print("The logistic regression accuracy {}".format(acc_log))

#creation of random forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print("The random forest accuracy {}".format(acc_random_forest))

# Representation of results by confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(Y_test, Y_prediction ), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for SVM')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


