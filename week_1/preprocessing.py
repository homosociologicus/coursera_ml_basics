import re

import pandas as pd


df = pd.read_csv('titanic.csv', index_col='PassengerId')
n_pass = df.shape[0]

# looking for the most popular woman's name
women = df[df.Sex == 'female']
names = dict()

# dropping brackets while processing names
for person in women.Name:
    for word in re.sub('[()]', '', person).split():
        names[word] = names.get(word, 0) + 1

answer = sorted([(value, key) for key, value in names.items()], reverse=True)

# printing all
print(
    f'''Passengers by sex:
{pd.value_counts(df.Sex)}

Survived, % of all:
{100 * pd.value_counts(df.Survived == 1) / n_pass}

First class, % of all:
{100 * pd.value_counts(df.Pclass == 1) / n_pass}

Mean age: {df.Age.mean()}

Median age: {df.Age.median()}

Correlation between numbers of siblings/spouses and parents/children: {df.SibSp.corr(df.Parch)}

5 most popular woman's names:
''',
    *answer[:5]
)
