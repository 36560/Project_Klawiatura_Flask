import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from joblib import dump, load


def convert_to_int(word):
    logins = {0: 0, 'Kinga': 1, 'Krystian': 2, 'Patryk': 3}
    return logins[word]

# Load data
data = pd.read_csv('short_plus_all.csv')
data.shape  # rows, cols

# Delete row with 0 values in time_to_next
listIndex = (data[data['time_to_next'] == 0].index.values)

print(listIndex)

for i in listIndex:
    data = data.drop(i)  # delete row

print(data.size)
print("Removing null values...")
data = data.dropna()
print(data.size)
# data
data.pop('id')
X = data.drop('login', axis=1)  # samples

y = data['login']  # Labels
y = y.apply(lambda x: convert_to_int(x))
y

print("Learning...")

kf = KFold(n_splits=2, shuffle=True)
model = RandomForestClassifier(n_estimators=114, bootstrap=True, max_features='sqrt', max_depth=39)


for train_index, test_index in kf.split(X):
    model.fit(X.values[train_index], y.values[train_index])
    score = model.score(X.values[train_index], y.values[train_index])
    print("Score dla train: ", score)
    score = model.score(X.values[test_index], y.values[test_index])
    print("Score dla train: ", score)

dump(model, 'model.joblib')