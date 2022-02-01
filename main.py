import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Loading the file into pd
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Columns = ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'
#  'Ticket' 'Fare' 'Cabin' 'Embarked']
print(train.head().to_string())
print("\n", "-" * 30)
mask = train[train['Age'].isnull()]  # empty ages
train_copy = train.copy()
train_copy['Age'].fillna(train_copy['Age'].median(), inplace=True)  # fills empty ages with median
train_copy['Embarked'].fillna('S', inplace=True)  # fills empty embarked with S (most common)
drop_column = ['PassengerId', 'Cabin', 'Ticket']
train_copy.drop(drop_column, axis=1, inplace=True)  # drop unusefull column
print(train_copy.isnull().sum())
print("\n", "-" * 30)
print(train_copy.head().to_string())
print("\n", "-" * 30)
print('Survival RATES :')
print("-" * 30)
women = train_copy.loc[train_copy.Sex == 'female']["Survived"]
rate_women = sum(women) / len(women)  # each survived is 1 so sum is women survived
men = train_copy.loc[train_copy.Sex == 'male']["Survived"]
rate_men = sum(men) / len(men)
print("Rate of women:", rate_women)
print("Rate of men:", rate_men)
print("-" * 30)
# cutting the age column to 5 pieces
ages = pd.cut(train_copy['Age'], 5, labels=['Child', 'Teen', 'Adult', 'Senior', 'Old'])
# [(0.34, 16.336] < (16.336, 32.252] < (32.252, 48.168] < (48.168, 64.084] < (64.084, 80.0]]
fares = pd.qcut(train_copy['Fare'], 4, labels=['Vlow', 'Low', 'Medium', ' High'])
# [(-0.001, 7.91] < (7.91, 14.454] < (14.454, 31.0] < (31.0, 512.329]]
train_copy['FareC'] = fares
train_copy['SortAges'] = ages
# Survived rate by age group
ages_rate = train_copy[['SortAges', 'Survived']].groupby(['SortAges'], as_index=False).mean()
print(ages_rate.head())
print("\n", "-" * 30)
cabin_rate = train_copy[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
print(cabin_rate.head())
print("\n", "-" * 30)
class_rate = train_copy[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
print(class_rate.head())
print("\n", "-" * 30)
fare_rate = train_copy[['FareC', 'Survived']].groupby(['FareC'], as_index=False).mean()
print(fare_rate.head())
print("\n", "-" * 30)

for dataset in [train_copy]:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in [train_copy]:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_copy[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
alone_rate = train_copy[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
print(alone_rate)
for dataset in [train_copy]:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_copy['Title'], train_copy['Sex'])

for dataset in [train_copy]:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

name_rate = train_copy[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
print(name_rate)
rate_pd = pd.DataFrame({
    'Category': ['Men', 'Women', 'Title', 'Fare', 'Embarked', 'Age', 'Class'],
    'SurvivalR': [rate_men, rate_women, name_rate[['Title', 'Survived']].to_string(index=False, header=None),
                  fare_rate.to_string(index=False, header=None),
                  cabin_rate.to_string(index=False, header=None), ages_rate.to_string(index=False, header=None),
                  class_rate.to_string(index=False, header=None)]

})
print("Sibsp rate", train_copy[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean())
print("Parch rate", train_copy[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean())
print("-" * 30)
print(rate_pd.to_markdown())
train_copy.drop(['Name'], axis=1, inplace=True)
train_copy.drop(['Parch', 'SibSp', 'FamilySize', 'Fare', 'Age'], axis=1, inplace=True)
#  Replacing data inside a row:
#  dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train_copy['Sex'] = train_copy['Sex'].map({'female': 'F', 'male': 'M'}).astype(str)
print("\n", train_copy.head(10).to_string())

name_dict = {'M': rate_men, 'F': rate_women}
for rates in [name_rate, fare_rate, cabin_rate, ages_rate, class_rate]:
    for i in range(0, len(rates)):
        name_dict[rates.iloc[i][0]] = rates.iloc[i][1]
for i in range(1, 4):
    name_dict[int(i)] = name_dict.pop(float(i))
print(name_dict)
train_copy2 = train_copy.copy()
for i in name_dict.keys():
    train_copy2.replace(to_replace=i,
                        value=name_dict[i],
                        inplace=True)

train_copy2['IsAlone'] = train_copy['IsAlone']
train_copy2.replace(to_replace=0,
                    value=0.505650,
                    inplace=True)
train_copy2.replace(to_replace=1,
                    value=0.303538,
                    inplace=True)
train_copy2['Survived'] = train_copy['Survived']
train_copy2['Average'] = (train_copy2[list(train_copy2.columns)].sum(axis=1) - 1) / 7
print(train_copy2.head().to_string())
train_copy2['Guess'] = 0
train_copy2.loc[train_copy2['Average'] > 0.38, 'Guess'] = 1
comparison_column = np.where(train_copy2["Survived"] == train_copy2["Guess"], 1, 0).sum()
print('rows that match:', comparison_column)
print(train_copy2[['Survived', 'Average', 'Guess']].head())

test_copy = test.copy()
test_copy.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
test_copy['Age'].fillna(test_copy['Age'].median(), inplace=True)
test_copy['Embarked'].fillna('S', inplace=True)
test_copy['Fare'].fillna(7.5, inplace=True)
ages_test = pd.cut(test_copy['Age'], 5, labels=['Child', 'Teen', 'Adult', 'Senior', 'Old'])
fares_test = pd.qcut(test_copy['Fare'], 4, labels=['Vlow', 'Low', 'Medium', ' High'])
test_copy['FareC'] = fares_test
test_copy['SortAges'] = ages_test
for dataset in [test_copy]:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in [test_copy]:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
test['IsAlone'] = test_copy['IsAlone']
for dataset in [test_copy]:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in [test_copy]:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

test_copy.drop(['Name'], axis=1, inplace=True)
test_copy.drop(['Parch', 'SibSp', 'FamilySize', 'Fare', 'Age'], axis=1, inplace=True)
test_copy['Sex'] = test_copy['Sex'].map({'female': 'F', 'male': 'M'}).astype(str)
for i in name_dict.keys():
    test_copy.replace(to_replace=i,
                      value=name_dict[i],
                      inplace=True)

test_copy['IsAlone'] = test['IsAlone']
test_copy.replace(to_replace=0,
                  value=0.505650,
                  inplace=True)
test_copy.replace(to_replace=1,
                  value=0.303538,
                  inplace=True)
test_copy.drop(['PassengerId'], axis=1, inplace=True)
test_copy['Average'] = (test_copy[list(test_copy.columns)].sum(axis=1)) / 7
test_copy['Guess'] = 0
test_copy.loc[test_copy['Average'] > 0.4, 'Guess'] = 1
print(test_copy.head().to_string())
test_copy.drop(['Pclass', 'Sex', 'Embarked', 'FareC', 'SortAges', 'IsAlone', 'Title'], axis=1, inplace=True)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_copy['Guess']
})
submission.to_csv('submission.csv', index=False)
test_copy['PassengerId'] = test['PassengerId']
test_copy.drop('Guess', axis=1, inplace=True)
print(test_copy.head())
train_last = train_copy2[['Survived', 'Average']]
print(train_last.head())
X_train = train_last.drop("Survived", axis=1)
Y_train = train_last["Survived"]
X_test = test_copy.drop("PassengerId", axis=1).copy()
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
submission2 = pd.DataFrame({
    "PassengerId": test_copy["PassengerId"],
    "Survived": Y_pred
})
submission2.to_csv('submission2.csv', index=False)
