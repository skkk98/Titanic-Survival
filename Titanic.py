import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('train.csv')
# print(data.info())
model = RandomForestClassifier(n_estimators=100)
freq_port = data.Embarked.dropna().mode()[0]
newd = [data]


for dataset in newd:
	dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

gender_map = {'female':1, 'male':0}
for dataset in newd:
	dataset['Sex'] = dataset['Sex'].map(gender_map)

embarked_map = {'S':0, 'C':1, 'Q':2}
for dataset in newd:
	dataset['Embarked'] = dataset['Embarked'].map(embarked_map)


'''grid = sns.FacetGrid(data, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()'''

guess_ages = np.zeros((2,3))



for dataset in newd:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)





X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
Y = data[['Survived']]
# print(data.describe())

model.fit(X,Y)

data1 = pd.read_csv('test.csv')
newd1 = [data1]

freq_port1 = data1.Embarked.dropna().mode()[0]

for dataset in newd1:
	dataset['Embarked'] = dataset['Embarked'].fillna(freq_port1)

for dataset in newd1:
	dataset['Sex'] = dataset['Sex'].map(gender_map)

for dataset in newd1:
	dataset['Embarked'] = dataset['Embarked'].map(embarked_map)


guess_ages1 = np.zeros((2,3))



for dataset in newd1:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages1[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages1[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

data1['Fare'].fillna(data1['Fare'].dropna().median(), inplace=True)

X_test = data1[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
'''print('-'*40)
print(X_test.isnull().values.any())
null_columns=X_test.columns[X_test.isnull().any()]
print('-'*40)
print(X_test[null_columns].isnull().sum())'''

# There is a null value present in Fare coloumn of Test dataset

Y_test = model.predict(X_test)
print(Y_test)

print(Y_test.shape)

acc_log = round(model.score(X, Y) * 100, 2)
print(acc_log)

submission = pd.DataFrame({
        "PassengerId": data1["PassengerId"],
        "Survived": Y_test
    })

submission.to_csv('final.csv', index=False)
